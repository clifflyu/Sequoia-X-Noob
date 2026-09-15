"""飞书通知属性测试。"""

import json
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

from sequoia_x.core.config import Settings
from sequoia_x.notify.feishu import FeishuNotifier


def make_settings(webhook_url: str = "https://example.com/default") -> Settings:
    return Settings(
        db_path="data/test.db",
        start_date="2024-01-01",
        feishu_webhook_url=webhook_url,
    )


# Feature: 每个策略的飞书通知只推送全市场排名前 3 只股票
@given(
    symbols=st.lists(
        st.text(min_size=6, max_size=6, alphabet="0123456789"),
        min_size=1, max_size=10, unique=True,
    )
)
@h_settings(max_examples=50)
def test_notification_contains_top_three_symbols(symbols: list[str]) -> None:
    """通知只保留策略排序靠前的三只股票。"""
    settings = make_settings()
    notifier = FeishuNotifier(settings)
    expected = notifier.select_symbols(symbols)

    with patch.object(notifier, "_get_stock_names", return_value={}):
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            notifier.send(symbols=symbols, strategy_name="TestStrategy")

    call_args = mock_post.call_args
    body = json.loads(call_args.kwargs.get("data") or call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs["data"])
    card_text = json.dumps(body)
    for symbol in expected:
        assert symbol in card_text
    for symbol in set(symbols) - set(expected):
        assert symbol not in card_text


def test_select_symbols_keeps_first_three() -> None:
    notifier = FeishuNotifier(make_settings())

    selected = notifier.select_symbols(
        ["000001", "000002", "000003", "000004", "600001", "600002", "600003", "600004"]
    )

    assert selected == ["000001", "000002", "000003"]


def test_notification_includes_trade_plan() -> None:
    """每只推送股票都应包含进场、止损和离场条件。"""
    notifier = FeishuNotifier(make_settings())
    with patch.object(notifier, "_get_stock_names", return_value={"000001": "平安银行"}):
        card = notifier._build_card(["000001"], "MaVolumeStrategy")

    content = card["card"]["elements"][2]["text"]["content"]
    assert "信号 K 线日期" in content
    assert "进场" in content
    assert "止损" in content
    assert "离场" in content


# Feature: sequoia-x-v2, Property 11: 飞书通知使用 ConfigManager 中的 Webhook URL
@given(
    webhook_url=st.from_regex(r"https://open\.feishu\.cn/open-apis/bot/v2/hook/[a-z0-9\-]{8,36}", fullmatch=True)
)
@h_settings(max_examples=50)
def test_notification_uses_config_url(webhook_url: str) -> None:
    """属性 11：send() 发出的 HTTP 请求目标 URL 应等于 settings.feishu_webhook_url。"""
    settings = make_settings(webhook_url=webhook_url)
    notifier = FeishuNotifier(settings)

    with patch.object(notifier, "_get_stock_names", return_value={}):
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            notifier.send(symbols=["000001"], strategy_name="Test", webhook_key="default")

    called_url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url")
    assert called_url == webhook_url


# Feature: sequoia-x-v2, Property 12: HTTP 失败时记录 ERROR 日志
@given(status_code=st.integers(min_value=400, max_value=599))
@h_settings(max_examples=50)
def test_http_failure_logs_error(status_code: int) -> None:
    """属性 12：非 200 响应时，send() 应记录 ERROR 级别日志，不抛出异常。"""
    import logging as _logging
    import sequoia_x.notify.feishu as feishu_module

    settings = make_settings()
    notifier = FeishuNotifier(settings)

    # feishu logger 设置了 propagate=False，需直接在其上挂 handler
    feishu_logger = _logging.getLogger(feishu_module.__name__)
    log_records: list[_logging.LogRecord] = []

    class _ListHandler(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            log_records.append(record)

    handler = _ListHandler(_logging.ERROR)
    feishu_logger.addHandler(handler)
    try:
        with patch.object(notifier, "_get_stock_names", return_value={}):
            with patch("requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=status_code, text="error")
                notifier.send(symbols=["000001"], strategy_name="Test")
    finally:
        feishu_logger.removeHandler(handler)

    assert any(r.levelno == _logging.ERROR for r in log_records)


# ── 未复权委托价的取数与容错 ──

SIGNAL_DATE = "2026-09-14"


class _OneRowResult:
    """只返回一行、模拟单日查询的结果集。"""

    error_code = "0"
    error_msg = "success"

    def __init__(self, row: list[str]) -> None:
        self._row = row
        self._consumed = False

    def next(self) -> bool:
        if self._consumed:
            return False
        self._consumed = True
        return True

    def get_row_data(self) -> list[str]:
        return self._row


class _StubMarketDataSession:
    """按 symbol 返回预设结果或抛预设异常的假行情会话。"""

    def __init__(self, behaviour) -> None:
        self._behaviour = behaviour

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def query_history_k_data_plus(self, code: str, fields: str, **kwargs):
        return self._behaviour(code.split(".")[-1])()


def _stub_engine():
    engine = MagicMock()
    engine._to_baostock_code.side_effect = lambda code: f"sh.{code}"
    engine.get_ohlcv.side_effect = lambda code: pd.DataFrame([{"date": SIGNAL_DATE}])
    return engine


def _session_returning(rows: dict[str, list[str]], failing: set[str] = frozenset()):
    """构造一个假会话工厂：failing 中的票抛连接异常，其余返回给定行。"""
    import requests as _requests

    def behaviour(symbol: str):
        if symbol in failing:
            def _fail():
                raise _requests.ConnectionError(
                    "Remote end closed connection without response"
                )
            return _fail
        return lambda: _OneRowResult(rows.get(symbol, []))

    return lambda: _StubMarketDataSession(behaviour)


def test_unadjusted_price_failure_is_isolated_per_symbol() -> None:
    """一只票取价失败，不能让同一张卡片其余票的委托价一起丢掉。

    改造前 try 包住整个 for 循环，任一票抛异常会让三只票全部退化成「未取得」。
    """
    symbols = ["000001", "000002", "000003"]
    notifier = FeishuNotifier(make_settings(), engine=_stub_engine())
    session = _session_returning(
        {symbol: [SIGNAL_DATE, "9.43", "9.24"] for symbol in symbols},
        failing={"000002"},
    )

    with patch("sequoia_x.data.baostock_client.MarketDataSession", session):
        notifier._load_unadjusted_signal_prices(symbols)

    assert notifier._unadjusted_signal_prices[("000001", SIGNAL_DATE)] == (9.43, 9.24)
    assert notifier._unadjusted_signal_prices[("000003", SIGNAL_DATE)] == (9.43, 9.24)
    assert ("000002", SIGNAL_DATE) not in notifier._unadjusted_signal_prices


def test_unadjusted_price_ignores_rows_from_other_dates() -> None:
    """信号日之外的行必须丢弃，避免把别的交易日价格当成委托价。"""
    notifier = FeishuNotifier(make_settings(), engine=_stub_engine())
    session = _session_returning({"000001": ["2026-09-11", "9.35", "9.22"]})

    with patch("sequoia_x.data.baostock_client.MarketDataSession", session):
        notifier._load_unadjusted_signal_prices(["000001"])

    assert notifier._unadjusted_signal_prices == {}


def test_card_shows_unadjusted_prices_instead_of_placeholder() -> None:
    """取到未复权价时，卡片应展示可下单的具体价位。"""
    notifier = FeishuNotifier(make_settings(), engine=_stub_engine())
    session = _session_returning({"000001": [SIGNAL_DATE, "9.43", "9.24"]})

    with patch("sequoia_x.data.baostock_client.MarketDataSession", session):
        with patch.object(notifier, "_get_stock_names", return_value={"000001": "平安银行"}):
            card = notifier._build_card(["000001"], "MaVolumeStrategy")

    content = card["card"]["elements"][2]["text"]["content"]
    assert "9.43" in content
    assert "9.24" in content
    assert "未取得" not in content


def test_production_incident_scenario_still_yields_prices(monkeypatch) -> None:
    """复现线上事故链路：baostock 被拉黑 + 东财按 IP 断连，卡片仍应给出委托价。

    这是本次故障的端到端回归：任一环节单独修好都不够，必须整条降级链走通。
    """
    import requests as _requests

    import sequoia_x.data.baostock_client as client_module

    class _BlacklistedBaostock:
        def __enter__(self):
            raise RuntimeError("baostock 登录失败: 黑名单用户，请与管理员联系")

        def __exit__(self, *args):
            return None

    class _BlockedAkshare:
        """东财被封时的表现：TLS 正常，但一发请求就被断连。"""

        @staticmethod
        def stock_zh_a_hist(**kwargs):
            raise _requests.ConnectionError(
                "('Connection aborted.', RemoteDisconnected('Remote end closed connection "
                "without response'))"
            )

    class _NoLockAkshareSession(client_module.AkshareSession):
        """去掉文件锁与 akshare 导入，保留真实的多源路由逻辑。"""

        def __enter__(self):
            self._ak = _BlockedAkshare()
            return self

        def __exit__(self, *args):
            self._ak = None
            return None

    class _OkResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    def _tencent_ok(url, params=None, **kwargs):
        prefixed = params["param"].split(",")[0]
        return _OkResponse(
            {
                "code": 0,
                "data": {
                    prefixed: {
                        "day": [[SIGNAL_DATE, "9.280", "9.400", "9.430", "9.240", "771871.000"]]
                    }
                },
            }
        )

    monkeypatch.setattr(client_module, "BaostockSession", _BlacklistedBaostock)
    monkeypatch.setattr(client_module, "AkshareSession", _NoLockAkshareSession)
    monkeypatch.setattr(client_module.requests, "get", _tencent_ok)
    monkeypatch.setattr(client_module.AkshareSession, "MIN_REQUEST_INTERVAL_SECONDS", 0)

    notifier = FeishuNotifier(make_settings(), engine=_stub_engine())
    with patch.object(notifier, "_get_stock_names", return_value={"000001": "平安银行"}):
        card = notifier._build_card(["000001"], "MaVolumeStrategy")

    content = card["card"]["elements"][2]["text"]["content"]
    assert "9.43" in content   # 进场：信号日未复权最高价
    assert "9.24" in content   # 止损：信号日未复权最低价
    assert "未取得" not in content
