from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import attribution as a
from generate_data import generate

T0 = datetime(2026, 3, 1, 12, 0)


@pytest.fixture()
def tiny(tmp_path: Path) -> Path:
    """Two users: u1 converts after search -> social -> email, u2 never converts."""
    touches = pd.DataFrame([
        ("u1", T0, "paid_search", "s"),
        ("u1", T0 + timedelta(hours=10), "paid_social", "p"),
        ("u1", T0 + timedelta(hours=20), "email", "e"),
        ("u2", T0, "display", "d"),
        ("u1", T0 - timedelta(days=40), "affiliate", "old"),  # outside 30-day lookback
    ], columns=["user_id", "touch_ts", "channel", "campaign_id"])
    clicks = pd.DataFrame([
        ("u1", T0, "home"),
        ("u1", T0 + timedelta(minutes=10), "browse"),
        ("u1", T0 + timedelta(minutes=50), "title"),   # 40 min gap -> new session
        ("u1", T0 + timedelta(minutes=55), "checkout"),
    ], columns=["user_id", "event_ts", "page"])
    conv = pd.DataFrame([("u1", T0 + timedelta(hours=21), "signup")],
                        columns=["user_id", "conversion_ts", "conversion_type"])
    touches.to_parquet(tmp_path / "touchpoints.parquet")
    clicks.to_parquet(tmp_path / "clickstream.parquet")
    conv.to_parquet(tmp_path / "conversions.parquet")
    return tmp_path


def _con(data: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    a.load(con, data)
    a.sessionize(con)
    a.build_journeys(con)
    a.rule_based_credit(con)
    return con


def test_sessionization_splits_on_gap(tiny):
    con = _con(tiny)
    sessions = con.execute("SELECT page_views, reached_checkout FROM fct_sessions ORDER BY session_start").fetchall()
    assert sessions == [(2, False), (2, True)]


def test_lookback_excludes_old_touches(tiny):
    con = _con(tiny)
    channels = {r[0] for r in con.execute("SELECT channel FROM int_journey_touches").fetchall()}
    assert channels == {"paid_search", "paid_social", "email"}


def test_rule_models_on_known_journey(tiny):
    con = _con(tiny)
    df = con.execute("SELECT channel, first_touch, last_touch, linear, position_based FROM fct_touch_credit").df()
    df = df.set_index("channel")
    assert df.loc["paid_search", "first_touch"] == 1.0
    assert df.loc["email", "last_touch"] == 1.0
    assert df["linear"].tolist() == pytest.approx([1 / 3] * 3)
    assert df.loc["paid_social", "position_based"] == pytest.approx(0.2)
    assert df.loc["email", "position_based"] == pytest.approx(0.4)


def test_time_decay_favors_recent_touches(tiny):
    con = _con(tiny)
    td = con.execute("SELECT channel, time_decay FROM fct_touch_credit").df().set_index("channel")["time_decay"]
    assert td["email"] > td["paid_social"] > td["paid_search"]


@pytest.mark.parametrize("model", a.RULE_MODELS)
def test_every_model_conserves_conversions(tmp_path, model):
    touches, events, conv = generate(n_users=1500, seed=3)
    touches.to_parquet(tmp_path / "touchpoints.parquet")
    events.to_parquet(tmp_path / "clickstream.parquet")
    conv.to_parquet(tmp_path / "conversions.parquet")
    con = _con(tmp_path)
    credited = con.execute(f"SELECT SUM({model}) FROM fct_touch_credit").fetchone()[0]
    journeys = con.execute("SELECT COUNT(DISTINCT conversion_id) FROM fct_touch_credit").fetchone()[0]
    assert credited == pytest.approx(journeys)


def test_markov_credit_sums_to_conversions(tmp_path):
    touches, events, conv = generate(n_users=1500, seed=4)
    touches.to_parquet(tmp_path / "touchpoints.parquet")
    events.to_parquet(tmp_path / "clickstream.parquet")
    conv.to_parquet(tmp_path / "conversions.parquet")
    con = _con(tmp_path)
    credit = a.markov_credit(con)
    assert credit.sum() == pytest.approx(conv["user_id"].nunique())
    assert (credit >= 0).all()
