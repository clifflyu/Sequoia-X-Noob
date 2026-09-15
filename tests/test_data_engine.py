"""数据引擎属性测试。"""

import sqlite3
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

import sequoia_x.data.baostock_client as client_module
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


# ── 降级批量同步的规模闸门 ──

_STALE_SYMBOLS = ("000001", "000002", "000003")


def _seed_stale_rows(engine: DataEngine, symbols=_STALE_SYMBOLS) -> None:
    """写入一批昨日以前的 K 线，让 sync_today_bulk 产生待同步任务。"""
    engine._upsert_daily(
        pd.DataFrame(
            [
                {"symbol": symbol, "date": "2025-01-02", "open": 1.0, "high": 1.0,
                 "low": 1.0, "close": 1.0, "volume": 1.0, "turnover": 1.0}
                for symbol in symbols
            ]
        )
    )


def _stub_session(provider: str, on_query):
    class _Session:
        def __init__(self) -> None:
            self.provider = provider

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def query_history_k_data_plus(self, code, fields, **kwargs):
            return on_query(self, code, fields)

    return _Session


def test_sync_today_bulk_skips_degraded_full_market(monkeypatch) -> None:
    """降级到 AKShare 且规模超阈值时跳过本轮，避免把出口 IP 打进东财黑名单。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine, _ = make_engine_in(tmp_dir)
        _seed_stale_rows(engine)
        monkeypatch.setattr(DataEngine, "AKSHARE_DEGRADED_MAX_TASKS", 2)
        queries: list[str] = []

        def on_query(session, code, fields):
            queries.append(code)
            raise AssertionError("降级且超阈值时不应发起任何查询")

        monkeypatch.setattr(
            client_module, "MarketDataSession", _stub_session("akshare", on_query)
        )

        assert engine.sync_today_bulk() == 0
        assert queries == []


def test_sync_today_bulk_allows_degraded_small_pool(monkeypatch) -> None:
    """降级但规模在阈值内时，仍应正常同步。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine, _ = make_engine_in(tmp_dir)
        _seed_stale_rows(engine, ("000001", "000002"))
        monkeypatch.setattr(DataEngine, "AKSHARE_DEGRADED_MAX_TASKS", 2)

        def on_query(session, code, fields):
            return client_module._RowsResult(
                fields.split(","),
                [["2026-09-15", "10", "11", "9", "10.5", "1000", "10500"]],
            )

        monkeypatch.setattr(
            client_module, "MarketDataSession", _stub_session("akshare", on_query)
        )

        assert engine.sync_today_bulk() == 2


def test_sync_today_bulk_stops_when_degraded_mid_loop(monkeypatch) -> None:
    """降级可能发生在循环中途（首个失败即永久切源），必须每轮复查数据源。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine, _ = make_engine_in(tmp_dir)
        _seed_stale_rows(engine)
        monkeypatch.setattr(DataEngine, "AKSHARE_DEGRADED_MAX_TASKS", 1)
        queried: list[str] = []

        def on_query(session, code, fields):
            queried.append(code)
            session.provider = "akshare"  # 首个请求之后即降级
            return client_module._RowsResult(
                fields.split(","),
                [["2026-09-15", "10", "11", "9", "10.5", "1000", "10500"]],
            )

        monkeypatch.setattr(
            client_module, "MarketDataSession", _stub_session("baostock", on_query)
        )

        # 阈值 1：第一只放行，第二只时 remaining=2 > 1 触发中止；
        # 已取到的部分仍应落库，而不是整轮丢弃。
        assert engine.sync_today_bulk() == 1
        assert len(queried) == 1
