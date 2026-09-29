"""
PROPCAST – Play rating: is a prop an optimal play or not?

Combines the model's probability for its preferred side with situational
evidence (teammate/opponent injuries, opponent defense, recent form, game
script, weather) and risk flags (own injury status, volatile low-volume role,
model far from the sportsbook, no real line) into:

    play_side    "over" / "under" (for anytime TD always "yes")
    play_rating  "optimal" / "lean" / "not_optimal"
    play_score   0-100
    play_factors list of {label, detail, impact}  impact: supports|against|risk|info

Evidence weights are percentage-point changes in how often a player beat his
own previous-5-game average, measured on 2023-2026 regular-season games
(~3.4k RB, ~7.6k WR/TE, ~1.5k QB rows). Signals that turned out to be noise
there (usage trends, rushing game script, totals) are deliberately left out,
and hot/cold streaks are scored the way the data says they behave (they
revert), not the way they "feel".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.ingestion.injury_status import is_out_status, ruled_out_player_games

# prop -> (family, stat column, usage column)
_PROP_FAMILY = {
    "rushing_yards": ("rush", "rushing_yards", "carries"),
    "rushing_attempts": ("rush", "carries", "carries"),
    "receiving_yards": ("recv", "receiving_yards", "targets"),
    "receptions": ("recv", "receptions", "targets"),
    "passing_yards": ("pass", "passing_yards", "attempts"),
    "passing_tds": ("pass", "passing_tds", "attempts"),
    "anytime_td": ("td", None, None),
}

# Measured effects, in percentage points of "beat his recent average".
_TEAMMATE_OUT = {"rush": 20.0, "recv": 9.0, "pass": 20.0, "td": 5.0}
_DEF_WEAK = {"rush": 3.2, "recv": 1.0, "pass": 5.0}     # opponent ranks 23-32
_DEF_STRONG = {"rush": -3.6, "recv": -1.0, "pass": -6.1}  # opponent ranks 1-10
_FORM_HOT = {"recv": -3.7}    # 3-game avg > 1.25x 8-game avg: tends to cool off
_FORM_COLD = {"recv": 4.9, "pass": 6.6}  # 3-game avg < 0.75x 8-game avg: bounce back
_FAVORED_BIG_RECV = -2.9      # receiver's team favored by 6+ (clock-killing script)
# Not backtestable yet (no history in the data), conventional small effects:
_OPP_CB1_OUT_RECV = 2.0
_HIGH_WIND = {"pass": -3.0, "recv": -3.0}

_QUESTIONABLE = ("questionable", "day to day", "day-to-day", "limited")


@dataclass
class PlayFactor:
    label: str
    detail: str
    impact: str  # supports | against | risk | info

    def as_dict(self) -> Dict[str, str]:
        return {"label": self.label, "detail": self.detail, "impact": self.impact}


@dataclass
class PlayRating:
    play_side: Optional[str]
    play_rating: str
    play_score: float
    play_summary: str
    play_factors: List[PlayFactor] = field(default_factory=list)
    increasing: List[str] = field(default_factory=list)
    decreasing: List[str] = field(default_factory=list)


class PlayContext:
    """Everything the rating needs, bulk-loaded once per request."""

    def __init__(self, db: Session, game_ids: Iterable[int], player_ids: Iterable[int]):
        self.game_ids = list(set(game_ids))
        self.player_ids = list(set(player_ids))
        self.games: Dict[int, Dict[str, Any]] = {}
        self.recent: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
        self.def_rank: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.depth: Dict[Tuple[int, int], Tuple[str, int]] = {}  # (player, game) -> (pos, rank)
        self.starters: Dict[Tuple[int, int, str], Tuple[int, str]] = {}  # (game, team, pos) -> (player, name)
        self.ruled_out: Set[Tuple[int, int]] = set()
        self.questionable: Dict[Tuple[int, int], str] = {}
        self.wind: Dict[int, Tuple[float, bool]] = {}
        if self.game_ids and self.player_ids:
            self._load(db)

    # ------------------------------------------------------------------
    @staticmethod
    def _in(prefix: str, values: List[int]) -> Tuple[str, Dict[str, int]]:
        params = {f"{prefix}{i}": v for i, v in enumerate(values)}
        return ", ".join(f":{k}" for k in params), params

    def _load(self, db: Session) -> None:
        g_in, g_params = self._in("g", self.game_ids)
        p_in, p_params = self._in("p", self.player_ids)

        for r in db.execute(text(f"""
            SELECT id, season, week, home_team_id, away_team_id, home_spread, game_total,
                   COALESCE(kickoff_time, created_at) AS kickoff
            FROM games WHERE id IN ({g_in})
        """), g_params).mappings():
            self.games[r["id"]] = dict(r)
        if not self.games:
            return
        seasons = sorted({g["season"] for g in self.games.values()})

        # --- recent regular-season games per player (before each game) ---
        rows = db.execute(text(f"""
            SELECT s.player_id, s.team_id, g.season, g.week, COALESCE(g.kickoff_time, g.created_at) AS kickoff,
                   COALESCE(s.rushing_yards, 0) AS rushing_yards, COALESCE(s.carries, 0) AS carries,
                   COALESCE(s.receiving_yards, 0) AS receiving_yards, COALESCE(s.receptions, 0) AS receptions,
                   COALESCE(s.targets, 0) AS targets, COALESCE(s.passing_yards, 0) AS passing_yards,
                   COALESCE(s.passing_tds, 0) AS passing_tds, COALESCE(s.attempts, 0) AS attempts,
                   COALESCE(s.rushing_tds, 0) + COALESCE(s.receiving_tds, 0) AS scrimmage_tds
            FROM player_game_stats s JOIN games g ON g.id = s.game_id
            WHERE s.player_id IN ({p_in}) AND g.game_type = 'REG' AND g.season >= :min_season
            ORDER BY s.player_id, kickoff
        """), {**p_params, "min_season": min(seasons) - 1}).mappings().all()
        by_player: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_player[r["player_id"]].append(dict(r))
        self._by_player = by_player

        # --- opponent defense ranks (per-game allowed, 1 = allows the least) ---
        def_rows = db.execute(text("""
            SELECT COALESCE(s.opponent_id, CASE WHEN s.team_id = g.home_team_id THEN g.away_team_id
                                                 WHEN s.team_id = g.away_team_id THEN g.home_team_id END) AS def_team,
                   g.season, g.id AS game_id,
                   SUM(COALESCE(s.rushing_yards, 0)) AS rush, SUM(COALESCE(s.passing_yards, 0)) AS pass
            FROM player_game_stats s JOIN games g ON g.id = s.game_id
            WHERE g.game_type = 'REG' AND g.season >= :min_season
            GROUP BY 1, 2, 3
        """), {"min_season": min(seasons) - 1}).mappings().all()
        agg: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0.0, 0.0, 0])
        for r in def_rows:
            if r["def_team"] is None:
                continue
            a = agg[(r["def_team"], r["season"])]
            a[0] += r["rush"]; a[1] += r["pass"]; a[2] += 1
        for season in {s for (_, s) in agg}:
            teams = [(t, v) for (t, s), v in agg.items() if s == season and v[2] > 0]
            for key, idx in (("rush", 0), ("pass", 1)):
                ordered = sorted(teams, key=lambda tv: tv[1][idx] / tv[1][2])
                for rank, (t, v) in enumerate(ordered, start=1):
                    self.def_rank.setdefault((t, season), {})[key] = rank
                    self.def_rank[(t, season)]["games"] = v[2]

        # --- depth charts for these games' weeks ---
        for r in db.execute(text(f"""
            SELECT g.id AS game_id, r.team_id, r.player_id, r.depth_chart_position AS pos,
                   r.depth_chart_rank AS rank, p.full_name
            FROM games g
            JOIN rosters r ON r.season = g.season AND r.week = g.week
                AND (r.team_id = g.home_team_id OR r.team_id = g.away_team_id)
            JOIN players p ON p.id = r.player_id
            WHERE g.id IN ({g_in}) AND r.depth_chart_position IN ('QB', 'RB', 'WR', 'TE', 'CB')
        """), g_params).mappings():
            key = (r["player_id"], r["game_id"])
            if key not in self.depth or r["rank"] < self.depth[key][1]:
                self.depth[key] = (r["pos"], r["rank"])
            if r["rank"] == 1:
                self.starters.setdefault((r["game_id"], r["team_id"], r["pos"]), (r["player_id"], r["full_name"]))

        # --- injuries ---
        self.ruled_out = ruled_out_player_games(db, self.game_ids)
        latest: Dict[Tuple[int, int], str] = {}
        for pid, gid, status in db.execute(text(f"""
            SELECT player_id, game_id, game_status FROM injuries
            WHERE game_id IN ({g_in}) ORDER BY report_date, id
        """), g_params).fetchall():
            latest[(pid, gid)] = status or ""
        for key, status in latest.items():
            s = status.strip().lower()
            if not is_out_status(status) and any(s.startswith(q) for q in _QUESTIONABLE):
                self.questionable[key] = status

        # --- weather ---
        try:
            for gid, wind, dome in db.execute(text(f"""
                SELECT game_id, wind_mph, is_dome FROM weather WHERE game_id IN ({g_in})
            """), g_params).fetchall():
                self.wind[gid] = (float(wind or 0.0), bool(dome))
        except Exception:  # weather table optional
            pass

    # ------------------------------------------------------------------
    def recent_games(self, player_id: int, game_id: int, n: int = 10) -> List[Dict[str, Any]]:
        g = self.games.get(game_id)
        rows = getattr(self, "_by_player", {}).get(player_id, [])
        if not g:
            return []
        prior = [r for r in rows if str(r["kickoff"]) < str(g["kickoff"])]
        return prior[-n:]

    def opp_team(self, team_id: Optional[int], game_id: int) -> Optional[int]:
        g = self.games.get(game_id)
        if not g or team_id is None:
            return None
        if team_id == g["home_team_id"]:
            return g["away_team_id"]
        if team_id == g["away_team_id"]:
            return g["home_team_id"]
        return None

    def defense_rank(self, def_team: Optional[int], game_id: int, kind: str) -> Tuple[Optional[int], Optional[int]]:
        """(rank, season used). Uses the current season once a team has 2+ games, else last season."""
        g = self.games.get(game_id)
        if not g or def_team is None:
            return None, None
        for season in (g["season"], g["season"] - 1):
            d = self.def_rank.get((def_team, season))
            if d and d.get("games", 0) >= (2 if season == g["season"] else 1) and kind in d:
                return d[kind], season
        return None, None


def _fmt(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def rate_play(
    ctx: PlayContext,
    *,
    player_id: int,
    player_name: str,
    position: str,
    team_id: Optional[int],
    game_id: int,
    prop_type: str,
    projection: Optional[float],
    market_line: Optional[float],
    has_real_line: bool,
    over_probability: Optional[float],
    market_over_prob: Optional[float] = None,
) -> PlayRating:
    family, stat, usage = _PROP_FAMILY.get(prop_type, (None, None, None))
    factors: List[PlayFactor] = []
    evidence_over = 0.0  # + pushes toward OVER, - toward UNDER
    risks = 0
    major_risk = False

    if over_probability is None or family is None:
        return PlayRating(None, "lean", 50.0, "Not enough data to rate this play.")

    is_td = prop_type == "anytime_td"
    side = "yes" if is_td else ("over" if over_probability >= 0.5 else "under")
    prob_side = over_probability if (is_td or side == "over") else 1.0 - over_probability
    line_txt = _fmt(market_line) if market_line is not None else "—"

    def add(label: str, detail: str, effect_over: float) -> None:
        nonlocal evidence_over
        evidence_over += effect_over
        toward_side = effect_over if (is_td or side == "over") else -effect_over
        factors.append(PlayFactor(label, detail, "supports" if toward_side > 0 else "against"))

    def risk(label: str, detail: str, major: bool = False) -> None:
        nonlocal risks, major_risk
        risks += 1
        major_risk = major_risk or major
        factors.append(PlayFactor(label, detail, "risk"))

    # --- model probability --------------------------------------------------
    if is_td:
        edge = over_probability - (market_over_prob if market_over_prob is not None else 0.5)
        factors.append(PlayFactor(
            "Model probability",
            f"Model gives {over_probability:.0%} to score vs {(market_over_prob or 0.5):.0%} implied by the odds.",
            "supports" if edge > 0 else "against",
        ))
    else:
        factors.append(PlayFactor(
            "Model probability",
            f"Model projects {_fmt(projection or 0)} vs line {line_txt}: {prob_side:.0%} chance of the {side.upper()}.",
            "supports" if prob_side >= 0.55 else "info",
        ))

    # --- teammate / opponent injuries (largest measured effect) -------------
    my_pos, my_rank = ctx.depth.get((player_id, game_id), (position, None))
    group = "WR" if position in ("WR", "TE") and family in ("recv", "td") else position
    starter = ctx.starters.get((game_id, team_id, group)) if team_id is not None else None
    promoted = bool(starter and starter[0] != player_id and (starter[0], game_id) in ctx.ruled_out)
    if promoted:
        add(
            "Starter ruled out",
            f"{starter[1]} ({group}1) is out: backups in this spot beat their recent average "
            f"~{_TEAMMATE_OUT[family]:.0f} pts more often.",
            _TEAMMATE_OUT[family],
        )
    if family in ("recv", "td") and position in ("WR", "TE", "RB"):
        qb1 = ctx.starters.get((game_id, team_id, "QB")) if team_id is not None else None
        if qb1 and (qb1[0], game_id) in ctx.ruled_out:
            factors.append(PlayFactor("Backup QB starting", f"{qb1[1]} is out; historically a wash for pass catchers.", "info"))
    opp = ctx.opp_team(team_id, game_id)
    if family == "recv" and opp is not None:
        cb1 = ctx.starters.get((game_id, opp, "CB"))
        if cb1 and (cb1[0], game_id) in ctx.ruled_out:
            add("Opposing CB1 out", f"{cb1[1]} is out for the defense.", _OPP_CB1_OUT_RECV)

    # --- own injury status --------------------------------------------------
    q = ctx.questionable.get((player_id, game_id))
    if q:
        risk("Injury designation", f"{player_name} is listed {q}: role and snap count are uncertain.", major=True)

    # --- opponent defense ---------------------------------------------------
    kind = "rush" if family == "rush" or (family == "td" and position == "RB") else "pass"
    rank, season_used = ctx.defense_rank(opp, game_id, kind)
    fam_key = family if family != "td" else ("rush" if kind == "rush" else "recv")
    if rank is not None:
        what = "run" if kind == "rush" else "pass"
        when = "" if season_used == ctx.games[game_id]["season"] else " (last season)"
        if rank >= 23:
            add("Weak defense", f"Opponent ranks #{rank} of 32 vs the {what}{when} (allows the most).", _DEF_WEAK[fam_key])
        elif rank <= 10:
            add("Tough defense", f"Opponent ranks #{rank} of 32 vs the {what}{when} (allows the least).", _DEF_STRONG[fam_key])
        else:
            factors.append(PlayFactor("Average defense", f"Opponent ranks #{rank} of 32 vs the {what}{when}.", "info"))

    # --- recent games -------------------------------------------------------
    recent = ctx.recent_games(player_id, game_id, n=10)
    if not is_td and stat:
        vals = [float(r[stat]) for r in recent]
        if len(vals) < 3:
            risk("Limited history", f"Only {len(vals)} recent regular-season games on file.")
        else:
            if market_line is not None:
                hits = sum(v > market_line for v in vals)
                factors.append(PlayFactor(
                    "Hit rate",
                    f"Went over {line_txt} in {hits} of his last {len(vals)} games "
                    f"(last 5 avg {_fmt(np.mean(vals[-5:]))}).",
                    "info",
                ))
            if len(vals) >= 6:
                l3, l8 = np.mean(vals[-3:]), np.mean(vals[-8:])
                if l8 > 0 and l3 > 1.25 * l8 and family in _FORM_HOT:
                    add("Recent spike", f"Last 3 avg {_fmt(l3)} vs 8-game avg {_fmt(l8)}: spikes like this usually cool off.",
                        _FORM_HOT[family])
                elif l8 > 0 and l3 < 0.75 * l8 and family in _FORM_COLD:
                    add("Due to bounce back", f"Last 3 avg {_fmt(l3)} vs 8-game avg {_fmt(l8)}: slumps like this usually recover.",
                        _FORM_COLD[family])

            # Volatile, low-volume role (e.g. a backup RB whose line swings on 2-3 touches).
            use = [float(r[usage]) for r in recent[-5:]] if usage else []
            low_volume = (
                (family == "rush" and position == "RB" and use and np.mean(use) < 8)
                or (family == "recv" and use and np.mean(use) < 4)
            )
            cv = float(np.std(vals[-8:]) / np.mean(vals[-8:])) if np.mean(vals[-8:]) > 0 else 0.0
            # Not flagged when the starter ahead of him is out: his old
            # low-volume games no longer describe his role this week.
            if low_volume and cv > 0.8 and not promoted:
                risk(
                    "Volatile role",
                    f"Averaging {_fmt(np.mean(use))} {usage} over his last {len(use)} games with big swings "
                    f"({', '.join(_fmt(v) for v in vals[-5:])}).",
                )

    # --- game script / weather ----------------------------------------------
    g = ctx.games.get(game_id, {})
    if family == "recv" and g.get("home_spread") is not None and team_id is not None:
        team_spread = g["home_spread"] if team_id == g["home_team_id"] else -g["home_spread"]
        if team_spread <= -6:
            add("Big favorite", f"Team favored by {_fmt(-team_spread)}: leads mean more running late.", _FAVORED_BIG_RECV)
    wind, dome = ctx.wind.get(game_id, (0.0, True))
    if family in _HIGH_WIND and not dome and wind >= 15:
        add("High wind", f"{_fmt(wind)} mph wind forecast.", _HIGH_WIND[family])

    # --- market sanity ------------------------------------------------------
    if is_td and not has_real_line:
        # Anytime TD odds aren't pulled (Odds API budget), so the odds shown are
        # a -110 placeholder and the "edge" above is measured against nothing
        # real. Never call that optimal or use it in play of the week.
        risk(
            "No sportsbook odds",
            "Anytime TD odds aren't pulled from the books; the odds shown are a placeholder.",
            major=True,
        )
    if not is_td:
        if not has_real_line:
            risk("No sportsbook line", "Line shown is generated, not a real book's number.")
        elif projection is not None and market_line is not None and market_line > 0:
            gap = abs(projection - market_line)
            is_count = prop_type in ("rushing_attempts", "receptions", "passing_tds")
            big = gap >= (max(1.5, 0.5 * market_line) if is_count else max(10.0, 0.5 * market_line))
            if big:
                risk(
                    "Far from the market",
                    f"Model ({_fmt(projection)}) is {_fmt(gap)} away from the book ({line_txt}). "
                    "Books often price role or injury news the model doesn't see.",
                    major=True,
                )

    # --- verdict ------------------------------------------------------------
    aligned = evidence_over if (is_td or side == "over") else -evidence_over
    if is_td:
        edge = over_probability - (market_over_prob if market_over_prob is not None else 0.5)
        base = 50 + edge * 200
        optimal = edge >= 0.05 and aligned >= 0 and risks == 0
        not_optimal = edge < 0 or major_risk or risks >= 2 or aligned <= -5
    else:
        base = 50 + (prob_side - 0.5) * 100
        optimal = prob_side >= 0.56 and aligned >= 0 and risks == 0
        not_optimal = prob_side < 0.52 or major_risk or risks >= 2 or aligned <= -5
    score = float(np.clip(base + aligned - 12 * risks, 0, 100))
    rating = "optimal" if optimal else ("not_optimal" if not_optimal else "lean")

    side_txt = "YES" if is_td else f"{side.upper()} {line_txt}"
    top = [f.detail for f in factors if f.impact == "risk"][:1] if rating == "not_optimal" else \
          [f.label.lower() for f in factors if f.impact == "supports" and f.label != "Model probability"][:2]
    verdict = {"optimal": "Optimal", "lean": "Lean", "not_optimal": "Not optimal"}[rating]
    summary = f"{verdict}: {side_txt}" + (f" — {'; '.join(top)}" if top else "")

    increasing, decreasing = [], []
    for f in factors:
        if f.impact in ("supports", "against") and f.label != "Model probability":
            up = (f.impact == "supports") == (is_td or side == "over")
            (increasing if up else decreasing).append(f"{f.label}: {f.detail}")

    return PlayRating(side, rating, round(score, 1), summary, factors, increasing, decreasing)
