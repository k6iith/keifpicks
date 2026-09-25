import httpx
import logging
from datetime import datetime
from backend.db.database import SessionLocal
from backend.db.models import Player, Team, Roster, Game, Prediction
from backend.ingestion.name_utils import find_player_by_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("espn_sync")

def sync_espn_rosters_and_depth_charts():
    db = SessionLocal()
    try:
        # Map DB teams
        db_teams = {t.abbreviation: t for t in db.query(Team).all()}
        # Also map standard ESPN abbr to DB team if any difference
        abbr_map = {
            'WSH': 'WAS',
            'LAR': 'LA',
        }
        
        # 1. Fetch all NFL teams from ESPN
        teams_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
        resp = httpx.get(teams_url, timeout=15).json()
        espn_teams = resp['sports'][0]['leagues'][0]['teams']
        
        logger.info(f"Fetched {len(espn_teams)} teams from ESPN.")
        
        # Clear existing rosters for 2026 week 3 to rebuild clean active starter depth chart
        db.query(Roster).filter(Roster.season == 2026, Roster.week == 3).delete()
        db.commit()

        # Mark all players CUT/INA by default, will activate those found on ESPN active rosters
        # Actually better: update players found on ESPN rosters to ACT and set team_id
        
        active_player_ids = set()
        starter_roster_entries = []
        
        for t_entry in espn_teams:
            t_data = t_entry['team']
            raw_abbr = t_data['abbreviation']
            team_abbr = abbr_map.get(raw_abbr, raw_abbr)
            espn_team_id = t_data['id']
            
            db_team = db_teams.get(team_abbr)
            if not db_team:
                continue
            
            # Fetch depth chart for this team
            dc_url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{espn_team_id}/depthcharts"
            try:
                dc_resp = httpx.get(dc_url, timeout=10)
                if dc_resp.status_code == 200:
                    dc_data = dc_resp.json()
                    for group in dc_data.get('depthchart', []):
                        positions = group.get('positions', {})
                        for pos_k, pos_v in positions.items():
                            pos_abbr = pos_v.get('position', {}).get('abbreviation')
                            if pos_abbr not in ['QB', 'RB', 'WR', 'TE', 'PK', 'K']:
                                continue
                            
                            norm_pos = 'K' if pos_abbr == 'PK' else pos_abbr
                            athletes = pos_v.get('athletes', [])
                            
                            for rank_idx, ath in enumerate(athletes, start=1):
                                ath_name = ath.get('displayName') or ath.get('fullName')
                                if not ath_name:
                                    continue
                                
                                # Match player in DB. ESPN's displayName often carries a
                                # generational suffix ("James Cook III") that our other
                                # sources omit ("James Cook"), so fall back to a
                                # suffix-insensitive match before creating a new row —
                                # otherwise every sync spawns a duplicate Player whose
                                # predictions never see the real Player's market lines.
                                player = find_player_by_name(db, ath_name, team_ids=[db_team.id])
                                if not player:
                                    # Try fuzzy or create
                                    player = Player(
                                        gsis_id=f"ESPN_{ath.get('id')}",
                                        full_name=ath_name,
                                        position=norm_pos,
                                        team_id=db_team.id,
                                        jersey_number=int(ath.get('jersey')) if str(ath.get('jersey', '')).isdigit() else None,
                                        status='ACT',
                                        headshot_url=ath.get('headshot', {}).get('href') or f"https://a.espncdn.com/i/headshots/nfl/players/full/{ath.get('id')}.png"
                                    )
                                    db.add(player)
                                    db.flush()
                                else:
                                    player.team_id = db_team.id
                                    player.status = 'ACT'
                                    if ath.get('headshot', {}).get('href'):
                                        player.headshot_url = ath['headshot']['href']
                                
                                active_player_ids.add(player.id)
                                
                                # Add depth chart entry
                                roster_entry = Roster(
                                    player_id=player.id,
                                    team_id=db_team.id,
                                    season=2026,
                                    week=3,
                                    depth_chart_position=norm_pos,
                                    depth_chart_rank=rank_idx
                                )
                                db.add(roster_entry)
            except Exception as e:
                logger.warning(f"Failed to fetch depth chart for {team_abbr}: {e}")
                
            # Also fetch standard active roster for any athletes not in depth chart
            roster_url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{espn_team_id}/roster"
            try:
                r_resp = httpx.get(roster_url, timeout=10)
                if r_resp.status_code == 200:
                    r_data = r_resp.json()
                    for cat in r_data.get('athletes', []):
                        cat_pos = cat.get('position')
                        if cat_pos == 'injuredReserveOrOut':
                            for ath in cat.get('items', []):
                                p = find_player_by_name(db, ath.get('displayName'), team_ids=[db_team.id])
                                if p:
                                    p.status = 'RES'
                        elif cat_pos in ['offense', 'specialTeam']:
                            for ath in cat.get('items', []):
                                p_name = ath.get('displayName')
                                p_pos = ath.get('position', {}).get('abbreviation')
                                if p_pos in ['QB', 'RB', 'WR', 'TE', 'PK', 'K']:
                                    norm_pos = 'K' if p_pos == 'PK' else p_pos
                                    p = find_player_by_name(db, p_name, team_ids=[db_team.id])
                                    if p:
                                        p.team_id = db_team.id
                                        p.status = 'ACT'
                                        active_player_ids.add(p.id)
            except Exception as e:
                logger.warning(f"Failed to fetch roster for {team_abbr}: {e}")
                
        db.commit()
        logger.info(f"ESPN sync finished: {len(active_player_ids)} active skill players verified.")
        
        # Now regenerate Week 3 predictions based exclusively on ML model pipeline
        logger.info("Generating statistical ML predictions for 2026 Week 3 starters...")
        try:
            from backend.features.builder import build_current_week_features
            from backend.models.predictor import generate_predictions_for_features
            
            w3_features = build_current_week_features(db, season=2026, week=3)
            if not w3_features.empty:
                # Mark old predictions non-current
                w3_games = db.query(Game).filter(Game.season == 2026, Game.week == 3).all()
                w3_gids = [g.id for g in w3_games]
                db.query(Prediction).filter(Prediction.game_id.in_(w3_gids)).update({"is_current": False})
                db.commit()
                
                preds_created = generate_predictions_for_features(db, w3_features)
                logger.info(f"Generated {preds_created} statistical ML predictions for 2026 Week 3.")
            else:
                logger.warning("No Week 3 features generated for active starters.")
        except Exception as e:
            logger.exception(f"Failed to generate ML predictions after ESPN sync: {e}")
            
    finally:
        db.close()

if __name__ == '__main__':
    sync_espn_rosters_and_depth_charts()
