"""行情源自动降级与 AKShare 字段归一化测试。"""

import sys

import pandas as pd
import pytest
import requests

import sequoia_x.data.baostock_client as client_module
from sequoia_x.data.baostock_client import (
    AkshareSession,
    BaostockSession,
    MarketDataSession,
    _RowsResult,
    _TencentQuoteSource,
)


@pytest.fixture(autouse=True)
def _disable_throttle(monkeypatch) -> None:
    """节流在测试里只会白等，统一关掉。"""
    monkeypatch.setattr(AkshareSession, "MIN_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(_TencentQuoteSource, "MIN_REQUEST_INTERVAL_SECONDS", 0)


class _FailedResult:
    error_code = "100"
    error_msg = "Baostock unavailable"


class _FakeBaostockSession:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def query_history_k_data_plus(self, *args, **kwargs):
        return _FailedResult()


class _FakeAkshareSession:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def query_history_k_data_plus(self, *args, **kwargs):
        return _RowsResult(["date", "close"], [["2026-09-14", "10.5"]])


def test_market_data_session_falls_back_after_baostock_result_error(monkeypatch) -> None:
    monkeypatch.setattr(client_module, "BaostockSession", _FakeBaostockSession)
    monkeypatch.setattr(client_module, "AkshareSession", _FakeAkshareSession)

    with MarketDataSession() as session:
        result = session.query_history_k_data_plus("sz.000001")
        assert session.provider == "akshare"
        assert result.next()
        assert result.get_row_data() == ["2026-09-14", "10.5"]


def test_akshare_history_normalizes_volume_from_lots_to_shares() -> None:
    class FakeAkshare:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            assert kwargs["adjust"] == "hfq"
            return pd.DataFrame(
                [{"日期": "2026-09-14", "开盘": 10, "收盘": 10.5, "成交量": 123}]
            )

    session = AkshareSession()
    session._ak = FakeAkshare()
    result = session.query_history_k_data_plus(
        "sz.000001", "date,open,close,volume", start_date="2026-09-14",
        end_date="2026-09-14", frequency="d", adjustflag="1",
    )

    assert result.next()
    assert result.get_row_data() == ["2026-09-14", "10", "10.5", "12300.0"]


# ── 腾讯不复权兜底源 ──


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _tencent_payload(rows: list[list[str]], *, key: str = "day") -> dict:
    """构造腾讯 ``fqkline/get`` 的返回体。"""
    return {"code": 0, "data": {"sh600000": {key: rows}}}


def _explode(*args, **kwargs):
    raise AssertionError("该路径不应访问腾讯")


def _patch_tencent(monkeypatch, rows: list[list[str]], *, key: str = "day") -> None:
    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: _FakeResponse(_tencent_payload(rows, key=key))
    )


def test_tencent_maps_open_close_high_low_in_declared_order(monkeypatch) -> None:
    """腾讯每行是「日期, 开, 收, 高, 低, 量」——错位会让 high/low 互换。

    这条是最关键的回归测试：一旦按 baostock 的「开高低收」顺序解析，
    卡片就会把止损价当成进场价推出去。数据取自 600000 在 2026-09-14 的真实返回。
    """
    _patch_tencent(monkeypatch, [["2026-09-14", "9.280", "9.400", "9.430", "9.240", "771871.000"]])

    session = AkshareSession()
    result = session.query_history_k_data_plus(
        "sh.600000", "date,open,high,low,close,volume",
        start_date="2026-09-14", end_date="2026-09-14", frequency="d", adjustflag="3",
    )

    assert result.next()
    assert result.get_row_data() == [
        "2026-09-14", "9.280", "9.430", "9.240", "9.400", "77187100.0",
    ]


def test_tencent_is_skipped_when_requested_field_is_unsupported(monkeypatch) -> None:
    """海龟策略要换手率，腾讯没有该字段，必须直接交给东财而不是返回错位数据。"""
    monkeypatch.setattr(requests, "get", _explode)

    class FakeAkshare:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            return pd.DataFrame(
                [{"日期": "2026-09-14", "收盘": 9.4, "成交量": 100, "换手率": 0.5}]
            )

    session = AkshareSession()
    session._ak = FakeAkshare()
    result = session.query_history_k_data_plus(
        "sh.600000", "close,volume,turn",
        start_date="2026-09-14", end_date="2026-09-14", frequency="d", adjustflag="3",
    )

    assert result.next()
    assert result.get_row_data() == ["9.4", "10000.0", "0.5"]


def test_tencent_rejects_adjusted_payload_on_unadjusted_path(monkeypatch) -> None:
    """腾讯若只返回复权数据，必须报错回退东财，绝不能把复权价当委托价。"""
    _patch_tencent(
        monkeypatch,
        [["2026-09-14", "129.125", "131.011", "131.097", "128.867", "1026696.000"]],
        key="hfqday",
    )
    eastmoney_calls = []

    class FakeAkshare:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            eastmoney_calls.append(kwargs)
            return pd.DataFrame([{"日期": "2026-09-14", "最高": 9.43, "最低": 9.24}])

    session = AkshareSession()
    session._ak = FakeAkshare()
    result = session.query_history_k_data_plus(
        "sh.600000", "date,high,low",
        start_date="2026-09-14", end_date="2026-09-14", frequency="d", adjustflag="3",
    )

    assert len(eastmoney_calls) == 1
    assert result.next()
    assert result.get_row_data() == ["2026-09-14", "9.43", "9.24"]


def test_tencent_filters_rows_outside_requested_window(monkeypatch) -> None:
    """区间外的行必须过滤掉，避免把别的交易日价格当成信号日价。"""
    _patch_tencent(
        monkeypatch,
        [
            ["2026-09-01", "9.13", "9.35", "9.36", "9.10", "1026696.000"],
            ["2026-09-11", "9.35", "9.26", "9.35", "9.22", "653273.000"],
        ],
    )

    session = AkshareSession()
    result = session.query_history_k_data_plus(
        "sh.600000", "date,high,low",
        start_date="2026-09-11", end_date="2026-09-11", frequency="d", adjustflag="3",
    )

    assert result.next()
    assert result.get_row_data() == ["2026-09-11", "9.35", "9.22"]
    assert not result.next()


def test_tencent_failure_falls_back_to_eastmoney(monkeypatch) -> None:
    """腾讯不可用时回退东财，而不是让调用方拿到空价。"""
    monkeypatch.setattr(client_module, "_call_with_backoff", lambda call, **kwargs: call())
    monkeypatch.setattr(requests, "get", _explode_connection)

    class FakeAkshare:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            assert kwargs["adjust"] == ""
            return pd.DataFrame([{"日期": "2026-09-14", "最高": 9.43, "最低": 9.24}])

    session = AkshareSession()
    session._ak = FakeAkshare()
    result = session.query_history_k_data_plus(
        "sh.600000", "date,high,low",
        start_date="2026-09-14", end_date="2026-09-14", frequency="d", adjustflag="3",
    )

    assert result.next()
    assert result.get_row_data() == ["2026-09-14", "9.43", "9.24"]


def _explode_connection(*args, **kwargs):
    raise requests.ConnectionError("Remote end closed connection without response")


# ── 复权口径红线 ──


def test_hfq_never_touches_tencent(monkeypatch) -> None:
    """后复权是唯一写入 stock_daily 的口径，腾讯 hfq 基准不同，绝不能混入。"""
    monkeypatch.setattr(requests, "get", _explode)

    class FakeAkshare:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            assert kwargs["adjust"] == "hfq"
            return pd.DataFrame([{"日期": "2026-09-14", "收盘": 10.5}])

    session = AkshareSession()
    session._ak = FakeAkshare()
    result = session.query_history_k_data_plus(
        "sz.000001", "date,close",
        start_date="2026-09-14", end_date="2026-09-14", frequency="d", adjustflag="1",
    )

    assert result.next()
    assert result.get_row_data() == ["2026-09-14", "10.5"]


def test_hfq_source_failure_raises_instead_of_substituting(monkeypatch) -> None:
    """后复权源挂掉时必须明确报错，不能悄悄换成别的口径。"""
    monkeypatch.setattr(client_module, "_call_with_backoff", lambda call, **kwargs: call())
    monkeypatch.setattr(requests, "get", _explode)

    class FakeAkshare:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            raise requests.ConnectionError("connection aborted")

    session = AkshareSession()
    session._ak = FakeAkshare()
    with pytest.raises(RuntimeError, match="后复权"):
        session.query_history_k_data_plus(
            "sz.000001", "date,close",
            start_date="2026-09-14", end_date="2026-09-14", frequency="d", adjustflag="1",
        )


def test_empty_history_returns_empty_result_instead_of_field_error() -> None:
    """停牌或非交易日返回空表，不该报「缺少字段」这种误导性错误。"""
    class FakeAkshare:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            return pd.DataFrame()

    session = AkshareSession()
    session._ak = FakeAkshare()
    result = session.query_history_k_data_plus(
        "sz.000001", "date,open,close",
        start_date="2026-09-14", end_date="2026-09-14", frequency="d", adjustflag="1",
    )

    assert result.fields == ["date", "open", "close"]
    assert not result.next()


# ── 会话熔断 ──


def test_consecutive_failures_trip_circuit_breaker(monkeypatch) -> None:
    """连续失败后熔断，避免对着已被限流的上游逐只空转。"""
    monkeypatch.setattr(client_module, "_call_with_backoff", lambda call, **kwargs: call())
    monkeypatch.setattr(AkshareSession, "CONSECUTIVE_FAILURE_LIMIT", 2)
    attempts = []

    class FakeAkshare:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            attempts.append(kwargs)
            raise requests.ConnectionError("aborted")

    session = AkshareSession()
    session._ak = FakeAkshare()
    for _ in range(2):
        with pytest.raises(requests.ConnectionError):
            session._request("stock_zh_a_hist")
    with pytest.raises(RuntimeError, match="熔断"):
        session._request("stock_zh_a_hist")

    assert len(attempts) == 2


# ── baostock 黑名单提示 ──


def _fake_baostock_module(error_code: str, error_msg: str):
    class _LoginResult:
        pass

    result = _LoginResult()
    result.error_code = error_code
    result.error_msg = error_msg

    class _Module:
        @staticmethod
        def login():
            return result

        @staticmethod
        def logout():
            return None

    return _Module


def test_blacklist_login_error_carries_actionable_hint(monkeypatch, tmp_path) -> None:
    """黑名单是服务端行为，报错要给出排查方向而不只是一句原始 error_msg。"""
    monkeypatch.setattr(BaostockSession, "_LOCK_PATH", tmp_path / "bs.lock")
    monkeypatch.setattr(BaostockSession, "_USAGE_DB_PATH", tmp_path / "usage.db")
    monkeypatch.setitem(
        sys.modules, "baostock", _fake_baostock_module("1", "黑名单用户，请与管理员联系")
    )

    with pytest.raises(RuntimeError) as excinfo:
        with BaostockSession():
            pass

    message = str(excinfo.value)
    assert "黑名单" in message
    assert "出口 IP" in message


def test_non_blacklist_login_error_has_no_blacklist_hint(monkeypatch, tmp_path) -> None:
    """普通失败不能被误报成黑名单，否则会误导排查方向。"""
    monkeypatch.setattr(BaostockSession, "_LOCK_PATH", tmp_path / "bs.lock")
    monkeypatch.setattr(BaostockSession, "_USAGE_DB_PATH", tmp_path / "usage.db")
    monkeypatch.setitem(sys.modules, "baostock", _fake_baostock_module("1", "网络错误"))

    with pytest.raises(RuntimeError) as excinfo:
        with BaostockSession():
            pass

    assert "出口 IP" not in str(excinfo.value)
