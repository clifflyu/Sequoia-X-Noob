"""行情源自动降级与 AKShare 字段归一化测试。"""

import pandas as pd

import sequoia_x.data.baostock_client as client_module
from sequoia_x.data.baostock_client import AkshareSession, MarketDataSession, _RowsResult


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
