"""Generate a synthetic clickstream + marketing touchpoint dataset.

The data mimics a subscription streaming service: users see ads across
channels, visit the site/app, and some sign up. A few "live event" campaigns
(think a big fight night or a playoff game) create short spikes in both
touchpoints and conversions, which is where attribution gets interesting.

Usage:
    python generate_data.py --users 20000 --out data/
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["paid_search", "paid_social", "display", "ctv", "email", "affiliate", "organic"]
# Relative chance that a touch comes from each channel.
CHANNEL_WEIGHTS = np.array([0.22, 0.20, 0.14, 0.08, 0.12, 0.06, 0.18])
# How much a touch from each channel lifts the odds of converting.
CHANNEL_LIFT = {
    "paid_search": 0.30, "paid_social": 0.18, "display": 0.06, "ctv": 0.12,
    "email": 0.22, "affiliate": 0.15, "organic": 0.10,
}
EVENTS = [
    # (campaign_id, event_date) - promos run in the 5 days before the event
    ("live_event_fight_night", datetime(2026, 3, 14)),
    ("live_event_playoff_game", datetime(2026, 4, 18)),
]
START = datetime(2026, 3, 1)
DAYS = 60


def generate(n_users: int, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    touches, events, conversions = [], [], []

    for uid in range(n_users):
        user_id = f"u{uid:06d}"
        n_touch = rng.poisson(3) + 1
        first_day = rng.integers(0, DAYS - 5)
        ts = START + timedelta(days=int(first_day), minutes=int(rng.integers(0, 1440)))
        score = -2.6
        for _ in range(n_touch):
            channel = rng.choice(CHANNELS, p=CHANNEL_WEIGHTS / CHANNEL_WEIGHTS.sum())
            campaign = f"{channel}_always_on"
            # Some paid touches belong to a live-event promo if they land in its window.
            for camp_id, ev_date in EVENTS:
                if channel in ("paid_social", "ctv", "paid_search") and ev_date - timedelta(days=5) <= ts <= ev_date:
                    if rng.random() < 0.6:
                        campaign = camp_id
                        score += 0.25  # event promos convert better
            touches.append((user_id, ts, channel, campaign))
            score += CHANNEL_LIFT[channel]

            # Each touch drives a short burst of page views (clickstream).
            n_views = rng.integers(1, 5)
            view_ts = ts + timedelta(seconds=int(rng.integers(5, 120)))
            for _ in range(n_views):
                events.append((user_id, view_ts, rng.choice(["home", "browse", "title", "pricing"])))
                view_ts += timedelta(seconds=int(rng.integers(20, 600)))

            ts += timedelta(hours=float(rng.exponential(30)))

        p = 1 / (1 + np.exp(-score))
        if rng.random() < p:
            conv_ts = ts + timedelta(hours=float(rng.exponential(6)))
            conversions.append((user_id, conv_ts, "signup"))
            events.append((user_id, conv_ts - timedelta(seconds=30), "checkout"))

    touch_df = pd.DataFrame(touches, columns=["user_id", "touch_ts", "channel", "campaign_id"])
    event_df = pd.DataFrame(events, columns=["user_id", "event_ts", "page"])
    conv_df = pd.DataFrame(conversions, columns=["user_id", "conversion_ts", "conversion_type"])
    return touch_df, event_df, conv_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    touches, events, conversions = generate(args.users, args.seed)
    touches.to_parquet(out / "touchpoints.parquet", index=False)
    events.to_parquet(out / "clickstream.parquet", index=False)
    conversions.to_parquet(out / "conversions.parquet", index=False)
    print(f"touchpoints={len(touches):,} clickstream={len(events):,} conversions={len(conversions):,}")


if __name__ == "__main__":
    main()
