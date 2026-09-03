"""
test_dynamic_universe.py
─────────────────────────
Unit tests for Point-in-Time Universe management and Parquet Partitioned Store.
Verifies survivorship-bias elimination and point-in-time slice correctness.
"""

import pandas as pd
import pytest
from pathlib import Path
import tempfile

from src.data.universe import (
    PointInTimeUniverse,
    TickerLifecycle,
    create_sample_pit_universe,
)
from src.data.partitioned_store import PartitionedTimeSeriesStore


def test_pit_universe_survivorship_inclusion():
    """Verify delisted assets are present BEFORE delisting and absent AFTER."""
    t_start = pd.Timestamp("2023-01-01", tz="UTC")
    t_delist = pd.Timestamp("2023-06-01", tz="UTC")
    t_after = pd.Timestamp("2023-08-01", tz="UTC")

    lc_survivor = TickerLifecycle("SURVIVOR", ipo_date=t_start)
    lc_delisted = TickerLifecycle("ENRON", ipo_date=t_start, delist_date=t_delist)

    pit = PointInTimeUniverse([lc_survivor, lc_delisted])

    # March 2023: both should be active
    active_march = pit.get_active_universe(pd.Timestamp("2023-03-01", tz="UTC"))
    assert "SURVIVOR" in active_march
    assert "ENRON" in active_march

    # July 2023: ENRON should be excluded (dead)
    active_july = pit.get_active_universe(t_after)
    assert "SURVIVOR" in active_july
    assert "ENRON" not in active_july


def test_pit_universe_ipo_exclusion():
    """Verify assets are excluded before their IPO date."""
    t_start = pd.Timestamp("2023-01-01", tz="UTC")
    t_ipo = pd.Timestamp("2023-06-01", tz="UTC")

    lc_ipo = TickerLifecycle("IPO_TICKER", ipo_date=t_ipo)
    pit = PointInTimeUniverse([lc_ipo])

    # Before IPO: not in universe
    assert "IPO_TICKER" not in pit.get_active_universe(t_start)

    # After IPO: in universe
    assert "IPO_TICKER" in pit.get_active_universe(pd.Timestamp("2023-07-01", tz="UTC"))


def test_filter_market_data_survivorship():
    """Test filtering DataFrame rows according to point-in-time lifecycles."""
    pit = create_sample_pit_universe("2023-01-01", "2023-12-31")

    df = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2023-03-01", tz="UTC"),
            pd.Timestamp("2023-09-01", tz="UTC"),
            pd.Timestamp("2023-03-01", tz="UTC"),
            pd.Timestamp("2023-09-01", tz="UTC"),
        ],
        "ticker": ["DEADCO", "DEADCO", "NEWCO", "NEWCO"],
        "close": [10.0, 5.0, 50.0, 55.0],
    })

    filtered = pit.filter_market_data(df)
    # DEADCO in March is valid, in Sept is invalid
    # NEWCO in March is invalid (pre-IPO), in Sept is valid
    assert len(filtered) == 2
    assert ("DEADCO" in filtered[filtered["timestamp"] == pd.Timestamp("2023-03-01", tz="UTC")]["ticker"].values)
    assert ("NEWCO" in filtered[filtered["timestamp"] == pd.Timestamp("2023-09-01", tz="UTC")]["ticker"].values)


def test_partitioned_store_read_write():
    """Test writing and reading partitioned Parquet datasets with predicate pushdown."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = PartitionedTimeSeriesStore(tmp_dir)

        df = pd.DataFrame({
            "timestamp": [
                pd.Timestamp("2023-01-15", tz="UTC"),
                pd.Timestamp("2023-02-15", tz="UTC"),
                pd.Timestamp("2023-03-15", tz="UTC"),
            ],
            "ticker": ["AAPL", "MSFT", "AAPL"],
            "open": [150.0, 240.0, 155.0],
            "high": [152.0, 245.0, 158.0],
            "low": [149.0, 239.0, 154.0],
            "close": [151.0, 244.0, 157.0],
            "volume": [1000.0, 2000.0, 1500.0],
        })

        store.write_dataset(df)

        # Query range
        res = store.read_range(
            start_date="2023-02-01",
            end_date="2023-03-31",
            tickers=["AAPL"],
        )
        assert len(res) == 1
        assert res.iloc[0]["ticker"] == "AAPL"
        assert res.iloc[0]["close"] == 157.0
