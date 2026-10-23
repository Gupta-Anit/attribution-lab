"""Clickstream sessionization and multi-touch attribution on DuckDB.

Pipeline:
  1. Load raw touchpoints, clickstream events and conversions.
  2. Sessionize clickstream (30-minute inactivity gap).
  3. Build conversion journeys: touches in the 30 days before each conversion.
  4. Score journeys with five rule-based models in SQL
     (first-touch, last-touch, linear, time-decay, position-based).
  5. Score channels with a first-order Markov chain removal-effect model.
  6. Write channel- and campaign-level credit tables.

Usage:
    python attribution.py --data data/ --db attribution.duckdb
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SESSION_GAP_MINUTES = 30
LOOKBACK_DAYS = 30
HALF_LIFE_HOURS = 7 * 24
RULE_MODELS = ["first_touch", "last_touch", "linear", "time_decay", "position_based"]


def load(con: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    for name in ("touchpoints", "clickstream", "conversions"):
        con.execute(
            f"CREATE OR REPLACE TABLE raw_{name} AS SELECT * FROM read_parquet(?)",
            [str(data_dir / f"{name}.parquet")],
        )


def sessionize(con: duckdb.DuckDBPyConnection, gap_minutes: int = SESSION_GAP_MINUTES) -> None:
    """Assign a session id to every clickstream event using an inactivity gap."""
    con.execute(f"""
        CREATE OR REPLACE TABLE stg_sessions AS
        WITH ordered AS (
            SELECT *,
                   LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) AS prev_ts
            FROM raw_clickstream
        ),
        flagged AS (
            SELECT *,
                   CASE WHEN prev_ts IS NULL
                          OR event_ts - prev_ts > INTERVAL {int(gap_minutes)} MINUTE
                        THEN 1 ELSE 0 END AS is_new_session
            FROM ordered
        )
        SELECT user_id, event_ts, page,
               user_id || '-' || SUM(is_new_session) OVER (
                   PARTITION BY user_id ORDER BY event_ts
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_id
        FROM flagged
    """)
    con.execute("""
        CREATE OR REPLACE TABLE fct_sessions AS
        SELECT session_id, user_id,
               MIN(event_ts) AS session_start,
               MAX(event_ts) AS session_end,
               COUNT(*) AS page_views,
               BOOL_OR(page = 'checkout') AS reached_checkout
        FROM stg_sessions
        GROUP BY ALL
    """)


def build_journeys(con: duckdb.DuckDBPyConnection, lookback_days: int = LOOKBACK_DAYS) -> None:
    """One row per (conversion, touch) with the touch position inside the journey."""
    con.execute(f"""
        CREATE OR REPLACE TABLE int_journey_touches AS
        WITH conv AS (
            SELECT user_id, conversion_ts,
                   user_id || '@' || CAST(conversion_ts AS VARCHAR) AS conversion_id
            FROM raw_conversions
        ),
        joined AS (
            SELECT c.conversion_id, c.user_id, c.conversion_ts,
                   t.touch_ts, t.channel, t.campaign_id
            FROM conv c
            JOIN raw_touchpoints t
              ON t.user_id = c.user_id
             AND t.touch_ts <= c.conversion_ts
             AND t.touch_ts >  c.conversion_ts - INTERVAL {int(lookback_days)} DAY
        )
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY conversion_id ORDER BY touch_ts) AS pos,
               COUNT(*)     OVER (PARTITION BY conversion_id)                   AS n_touches,
               DATE_DIFF('second', touch_ts, conversion_ts) / 3600.0            AS hours_before
        FROM joined
    """)


def rule_based_credit(con: duckdb.DuckDBPyConnection, half_life_hours: float = HALF_LIFE_HOURS) -> None:
    """Fractional credit per touch for each rule-based model. Credits sum to 1 per conversion."""
    con.execute(f"""
        CREATE OR REPLACE TABLE fct_touch_credit AS
        WITH w AS (
            SELECT *, POW(2, -hours_before / {float(half_life_hours)}) AS decay_w
            FROM int_journey_touches
        )
        SELECT conversion_id, user_id, touch_ts, channel, campaign_id, pos, n_touches,
               CASE WHEN pos = 1 THEN 1.0 ELSE 0.0 END                     AS first_touch,
               CASE WHEN pos = n_touches THEN 1.0 ELSE 0.0 END             AS last_touch,
               1.0 / n_touches                                              AS linear,
               decay_w / SUM(decay_w) OVER (PARTITION BY conversion_id)     AS time_decay,
               CASE
                   WHEN n_touches = 1 THEN 1.0
                   WHEN n_touches = 2 THEN 0.5
                   WHEN pos = 1 OR pos = n_touches THEN 0.4
                   ELSE 0.2 / (n_touches - 2)
               END                                                          AS position_based
        FROM w
    """)


def markov_credit(con: duckdb.DuckDBPyConnection, lookback_days: int = LOOKBACK_DAYS) -> pd.Series:
    """First-order Markov chain attribution using the removal effect.

    Paths include non-converting users (their touches end in 'null'), which is
    what lets the model see which channels actually change the odds of converting.
    Returns credited conversions per channel, summing to total conversions.
    """
    paths = con.execute(f"""
        WITH conv AS (SELECT user_id, MIN(conversion_ts) AS conversion_ts FROM raw_conversions GROUP BY 1),
        t AS (
            SELECT t.user_id, t.touch_ts, t.channel, c.conversion_ts
            FROM raw_touchpoints t LEFT JOIN conv c USING (user_id)
            WHERE c.conversion_ts IS NULL
               OR (t.touch_ts <= c.conversion_ts
                   AND t.touch_ts > c.conversion_ts - INTERVAL {int(lookback_days)} DAY)
        )
        SELECT user_id,
               LIST(channel ORDER BY touch_ts) AS path,
               BOOL_OR(conversion_ts IS NOT NULL) AS converted
        FROM t GROUP BY user_id
    """).df()

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for path, converted in zip(paths["path"], paths["converted"]):
        states = ["start", *list(path), "conv" if converted else "null"]
        for a, b in zip(states[:-1], states[1:]):
            counts[a][b] += 1

    channels = sorted({s for row in counts.values() for s in row} - {"conv", "null"})
    total_conversions = int(paths["converted"].sum())
    base = _conversion_probability(counts, removed=None)
    if base == 0 or total_conversions == 0:
        return pd.Series(0.0, index=channels)

    removal = {ch: 1 - _conversion_probability(counts, removed=ch) / base for ch in channels}
    total_removal = sum(removal.values())
    return pd.Series({ch: total_conversions * r / total_removal for ch, r in removal.items()})


def _conversion_probability(counts: dict[str, dict[str, int]], removed: str | None) -> float:
    """Probability of reaching 'conv' from 'start'. A removed channel sends all traffic to 'null'."""
    transient = sorted(s for s in counts if s not in ("conv", "null"))
    idx = {s: i for i, s in enumerate(transient)}
    n = len(transient)
    q = np.zeros((n, n))  # transient -> transient
    r = np.zeros(n)        # transient -> conv
    for a in transient:
        if a == removed:
            continue  # row of zeros: every path through this channel is absorbed by 'null'
        total = sum(counts[a].values())
        for b, c in counts[a].items():
            p = c / total
            if b == "conv":
                r[idx[a]] += p
            elif b != "null":
                q[idx[a], idx[b]] += p
    # Absorption probabilities: (I - Q) x = R
    x = np.linalg.solve(np.eye(n) - q, r)
    return float(x[idx["start"]])


def build_outputs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rule = con.execute(f"""
        SELECT channel, {", ".join(f"SUM({m}) AS {m}" for m in RULE_MODELS)}
        FROM fct_touch_credit GROUP BY channel
    """).df().set_index("channel")
    rule["markov"] = markov_credit(con)
    rule = rule.fillna(0.0).sort_values("linear", ascending=False).round(1)
    con.register("channel_credit_df", rule.reset_index())
    con.execute("CREATE OR REPLACE TABLE rpt_channel_credit AS SELECT * FROM channel_credit_df")
    con.execute("""
        CREATE OR REPLACE TABLE rpt_campaign_credit AS
        SELECT campaign_id, channel,
               ROUND(SUM(linear), 1)      AS linear_conversions,
               ROUND(SUM(time_decay), 1)  AS time_decay_conversions,
               COUNT(DISTINCT conversion_id) AS assisted_conversions
        FROM fct_touch_credit GROUP BY ALL ORDER BY linear_conversions DESC
    """)
    return rule


def run(data_dir: Path, db_path: str = ":memory:") -> pd.DataFrame:
    con = duckdb.connect(db_path)
    load(con, data_dir)
    sessionize(con)
    build_journeys(con)
    rule_based_credit(con)
    return build_outputs(con)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--db", default="attribution.duckdb")
    args = ap.parse_args()
    table = run(Path(args.data), args.db)
    print("Credited conversions by channel and model:\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
