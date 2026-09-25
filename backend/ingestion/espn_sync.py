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
                                        jersey_number=int(ath.get
