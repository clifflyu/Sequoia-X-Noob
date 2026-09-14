"""数据引擎属性测试。"""

import sqlite3
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine


def make_engine_in(tmp_dir: str) -> tuple[DataEngine, Settings]:
    """创建使用临时数据库的 DataEngine 实例。"""
    settings = Settings(
        db_path=str(Path(tmp_dir) / "test.db"),
        start_date="2024-01-01",
        feishu_webhook_url="https://example.com/hook",
    )
    engine = DataEngine(settings)
    return engine, settings


# Property 4: (symbol, date) 唯一约束防止重复写入
@given(
    symbol=st.text(min_size=6, max_size=6, alphabet="0123456789"),
    trade_date=st.dates(min_value=date(2024, 1, 1), max_value=date(2025, 12, 31)),
)
@h_settings(max_examples=50, deadline=None)
def test_unique_symbol_date_constraint(symbol: str, trade_date: date) -> None:
    """相同 (symbol, date) 插入两次，数据库中该组合记录数应保持为 1。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine, _ = make_engine_in(tmp_dir)
        row = {
            "symbol": symbol, "date": str(trade_date),
            "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
            "volume": 1000.0, "turnover": 10500.0,
        }
        df = pd.DataFrame([row])
        with sqlite3.connect(engine.db_path) as conn:
            df.to_sql("stock_daily", conn, if_exists="append", index=False, method="multi")
            try:
                df.to_sql("stock_daily", conn, if_exists="append", index=False, method="multi")
            except sqlite3.IntegrityError:
                pass
            count = conn.execute(
                "SELECT COUNT(*) FROM stock_daily WHERE symbol=? AND date=?",
                (symbol, str(trade_date)),
            ).fetchone()[0]
        assert count == 1


def test_incremental_upsert_does_not_remove_other_symbols() -> None:
    """指定股票池同步时，不能删除同日但不在该股票池中的历史数据。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine, _ = make_engine_in(tmp_dir)
        with sqlite3.connect(engine.db_path) as conn:
            conn.execute(
                "INSERT INTO stock_daily (symbol, date, close) VALUES (?, ?, ?)",
                ("600519", "2025-01-02", 1500.0),
            )
            conn.commit()

        engine._upsert_daily(
            pd.DataFrame(
                [{
                    "symbol": "000001", "date": "2025-01-02", "open": 10.0,
                    "high": 11.0, "low": 9.0, "close": 10.5,
                    "volume": 1000.0, "turnover": 10500.0,
                }]
            )
        )

        with sqlite3.connect(engine.db_path) as conn:
            other_symbol = conn.execute(
                "SELECT close FROM stock_daily WHERE symbol = ? AND date = ?",
                ("600519", "2025-01-02"),
            ).fetchone()
            updated_symbol = conn.execute(
                "SELECT close FROM stock_daily WHERE symbol = ? AND date = ?",
                ("000001", "2025-01-02"),
            ).fetchone()

        assert other_symbol == (1500.0,)
        assert updated_symbol == (10.5,)


def test_universe_limits_local_symbols() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine, _ = make_engine_in(tmp_dir)
        engine._upsert_daily(
            pd.DataFrame(
                [
                    {"symbol": "000001", "date": "2025-01-02", "open": 10.0, "high": 11.0,
                     "low": 9.0, "close": 10.5, "volume": 1000.0, "turnover": 10500.0},
                    {"symbol": "600519", "date": "2025-01-02", "open": 1500.0, "high": 1510.0,
                     "low": 1490.0, "close": 1505.0, "volume": 1000.0, "turnover": 1505000.0},
                ]
            )
        )

        engine.set_universe(["600519"])

        assert engine.get_local_symbols() == ["600519"]
