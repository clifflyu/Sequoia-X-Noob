"""Baostock 的串行访问保护。

官网明确禁止并发连接。所有 Baostock 请求必须经由 ``BaostockSession``，
它使用跨进程文件锁保证同一台机器上任意时刻只有一个登录会话，并将当日
请求数保存在独立 SQLite 文件中，避免重启后绕过限额。
"""

from __future__ import annotations

import fcntl
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import requests

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class BaostockDailyLimitExceeded(RuntimeError):
    """在安全余量内达到 Baostock 自然日调用上限。"""


class BaostockSession:
    """独占、限速且受日配额保护的 Baostock 会话。"""

    # 官网上限为 50,000；预留 10% 余量，防止其他未计量客户端造成误判。
    DAILY_REQUEST_LIMIT = 45_000
    MIN_REQUEST_INTERVAL_SECONDS = 0.2
    # 封禁由服务端控制并会自行解除，反复重试只会延长封禁，因此不做自动重试。
    _BLACKLIST_HINT = (
        "baostock 已把本机出口 IP 列入黑名单，通常由并发连接或短时高频请求触发。"
        "该封禁由服务端控制、会自行解除，请勿反复重试；可先等待恢复，"
        "并确认没有其他进程或机器共用同一出口 IP 访问 baostock。"
    )
    _LOCK_PATH = Path(tempfile.gettempdir()) / "sequoia-x-baostock.lock"
    _USAGE_DB_PATH = Path(tempfile.gettempdir()) / "sequoia-x-baostock-usage.sqlite3"

    def __init__(self) -> None:
        self._lock_file: Any | None = None
        self._bs: Any | None = None
        self._last_request_at = 0.0

    @staticmethod
    def _today() -> str:
        return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    def __enter__(self) -> BaostockSession:
        self._LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self._LOCK_PATH, "a+", encoding="utf-8")
        logger.info("等待 Baostock 独占访问锁")
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        try:
            import baostock as bs

            self._bs = bs
            login = self._request("login")
            if login.error_code != "0":
                if "黑名单" in login.error_msg:
                    raise RuntimeError(
                        f"baostock 登录失败: {login.error_msg} —— {self._BLACKLIST_HINT}"
                    )
                raise RuntimeError(f"baostock 登录失败: {login.error_msg}")
            return self
        except Exception:
            self._release_lock()
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if self._bs is not None:
                try:
                    self._request("logout", enforce_limit=False)
                except Exception as logout_error:
                    logger.warning(f"baostock 登出失败: {logout_error}")
        finally:
            self._bs = None
            self._release_lock()

    def _release_lock(self) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def _consume_request(self, *, enforce_limit: bool) -> None:
        """在持有全局锁时原子地记账，避免多个进程超限。"""
        with sqlite3.connect(self._USAGE_DB_PATH) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS baostock_request_usage "
                "(day TEXT PRIMARY KEY, requests INTEGER NOT NULL)"
            )
            day = self._today()
            row = conn.execute(
                "SELECT requests FROM baostock_request_usage WHERE day = ?", (day,)
            ).fetchone()
            used = row[0] if row else 0
            if enforce_limit and used >= self.DAILY_REQUEST_LIMIT:
                raise BaostockDailyLimitExceeded(
                    f"Baostock 当日请求已达安全上限 {self.DAILY_REQUEST_LIMIT}，已停止调用"
                )
            if used + 1 == int(self.DAILY_REQUEST_LIMIT * 0.8):
                logger.warning(
                    "Baostock 当日请求已使用 80%%（%s/%s），请避免额外手工调用",
                    used + 1,
                    self.DAILY_REQUEST_LIMIT,
                )
            conn.execute(
                "INSERT INTO baostock_request_usage(day, requests) VALUES (?, 1) "
                "ON CONFLICT(day) DO UPDATE SET requests = requests + 1",
                (day,),
            )
            conn.execute(
                "DELETE FROM baostock_request_usage WHERE day < date(?, '-7 days')", (day,)
            )
            conn.commit()

    def _request(self, method: str, *args: Any, enforce_limit: bool = True, **kwargs: Any) -> Any:
        if self._bs is None:
            raise RuntimeError("Baostock 会话尚未登录")
        self._consume_request(enforce_limit=enforce_limit)
        delay = self.MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
        if delay > 0:
            time.sleep(delay)
        result = getattr(self._bs, method)(*args, **kwargs)
        self._last_request_at = time.monotonic()
        return result

    def query_history_k_data_plus(self, *args: Any, **kwargs: Any) -> Any:
        return self._request("query_history_k_data_plus", *args, **kwargs)

    def query_stock_basic(self, *args: Any, **kwargs: Any) -> Any:
        return self._request("query_stock_basic", *args, **kwargs)


class _RowsResult:
    """兼容 Baostock ``ResultData`` 最小读取接口的内存结果集。"""

    error_code = "0"
    error_msg = "success"

    def __init__(self, fields: list[str], rows: list[list[str]]) -> None:
        self.fields = fields
        self._rows = rows
        self._index = -1

    def next(self) -> bool:
        self._index += 1
        return self._index < len(self._rows)

    def get_row_data(self) -> list[str]:
        return self._rows[self._index]


def _call_with_backoff(
    call: Callable[[], Any], *, label: str, attempts: int = 3, base_delay_seconds: int = 2
) -> Any:
    """短重试 + 指数退避，退避节奏（2s/4s）与 ``DataEngine.backfill`` 保持一致。

    最后一次尝试失败后不再等待，直接抛出，由调用方决定降级还是中止。
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                wait = base_delay_seconds ** (attempt + 1)
                logger.warning(f"{label} 第{attempt + 1}次失败: {exc}，{wait}s 后重试")
                time.sleep(wait)
    assert last_error is not None
    raise last_error


class _TencentQuoteSource:
    """腾讯行情兜底源，只提供不复权日线。

    东财 ``push2his`` 接口一旦触发风控就会按 IP 直接断连（TLS 握手正常，请求发出
    即被关闭），降级链需要一条不经过东财的退路。腾讯该接口无需登录鉴权，实测返回
    的不复权价格与新浪实时快照逐字段一致。

    刻意只服务「不复权且不落库」的场景：腾讯的后复权基准与 Baostock 不同，且比值
    随日期浮动、无法换算，一旦混入 ``stock_daily`` 会静默污染策略数据——涨跌停判断
    依赖的是 1.095/0.905 这类比值阈值，口径串了会直接产出假信号。
    """

    _ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    MIN_REQUEST_INTERVAL_SECONDS = 1.0
    REQUEST_TIMEOUT_SECONDS = 10.0
    MAX_ROWS = 640
    # 腾讯日线每行是「日期, 开, 收, 高, 低, 成交量(手)」——是开收高低，不是开高低收。
    # 顺序搞反会让 high/low 互换，卡片就会把止损价当成进场价推出去。
    _COLUMN_INDEX = {"date": 0, "open": 1, "close": 2, "high": 3, "low": 4, "volume": 5}
    # 腾讯不提供换手率与成交额；调用方要这些字段时只能走东财。
    SUPPORTED_FIELDS = frozenset(_COLUMN_INDEX)

    def __init__(self) -> None:
        self._last_request_at = 0.0

    @staticmethod
    def _market_prefix(symbol: str) -> str:
        """6/9 开头为沪市，4/8 开头为北交所，其余为深市。"""
        if symbol.startswith(("6", "9")):
            return "sh"
        if symbol.startswith(("4", "8")):
            return "bj"
        return "sz"

    def _fetch(self, prefixed: str, start_date: str, end_date: str) -> dict[str, Any]:
        delay = self.MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
        if delay > 0:
            time.sleep(delay)
        try:
            response = requests.get(
                self._ENDPOINT,
                params={"param": f"{prefixed},day,{start_date},{end_date},{self.MAX_ROWS},"},
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        finally:
            # 失败也计入冷却，避免对上游密集重试。
            self._last_request_at = time.monotonic()
        data = payload.get("data")
        if not isinstance(data, dict):
            # 该接口正常返回 ``{"data": {"<code>": {...}}}``；形态变了就当作无数据，
            # 让调用方回退东财，不要在这里重试三次才失败。
            return {}
        return data.get(prefixed) or {}

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        *,
        start_date: str,
        end_date: str,
        frequency: str,
        adjustflag: str,
    ) -> _RowsResult:
        if frequency != "d":
            raise ValueError("腾讯源当前仅支持日线")
        if adjustflag != "3":
            raise ValueError("腾讯源仅提供不复权日线（adjustflag=3）")
        requested = fields.split(",")
        # 腾讯不提供换手率与成交额，缺字段必须显式报错，不能返回错位数据。
        unsupported = [name for name in requested if name not in self._COLUMN_INDEX]
        if unsupported:
            raise RuntimeError(f"腾讯源不支持字段: {', '.join(unsupported)}")

        symbol = code.split(".")[-1]
        prefixed = f"{self._market_prefix(symbol)}{symbol}"
        node = _call_with_backoff(
            lambda: self._fetch(prefixed, start_date, end_date),
            label=f"腾讯日线[{symbol}]",
        )

        raw_rows = node.get("day")
        if raw_rows is None:
            # 兜底闸门：一旦请求参数被改坏，腾讯会改返回 qfqday/hfqday。复权价与不复权价
            # 相差一个数量级，宁可报错回退东财，也绝不能把复权价当成委托价推出去。
            if "qfqday" in node or "hfqday" in node:
                raise RuntimeError("腾讯仅返回了复权口径数据，拒绝用于不复权路径")
            # 停牌、非交易日或代码不存在：与本源其余行为一致，返回空结果而非报错。
            raw_rows = []

        rows: list[list[str]] = []
        for raw in raw_rows:
            if len(raw) < len(self._COLUMN_INDEX):
                continue
            day = str(raw[self._COLUMN_INDEX["date"]])[:10]
            # count 只是条数上限、区间才是主过滤条件，这里再兜一层防止取到区间外的价。
            if day < start_date or day > end_date:
                continue
            row: list[str] = []
            for name in requested:
                value = raw[self._COLUMN_INDEX[name]]
                if name == "volume":
                    # 腾讯返回手；本项目与 Baostock 一样统一存为股。
                    value = float(value) * 100
                elif name == "date":
                    value = day
                row.append(str(value))
            rows.append(row)
        rows.sort(key=lambda item: item[0])
        return _RowsResult(requested, rows)


class AkshareSession:
    """AKShare 的串行节流会话，避免对其底层公开数据源施压。

    东财接口触发风控后会直接断连（``RemoteDisconnected``）并持续数十分钟，因此这里
    除固定节流外还带两层保护：短重试扛住偶发抖动，连续失败熔断避免对着已被封禁的源
    逐只空转把整轮同步的时间耗光。
    """

    MIN_REQUEST_INTERVAL_SECONDS = 1.0
    # 连续失败达到该次数即认为上游已被限流，继续重试没有意义。
    CONSECUTIVE_FAILURE_LIMIT = 5
    REQUEST_TIMEOUT_SECONDS = 10.0
    LOCK_WAIT_SECONDS = 120.0
    LOCK_POLL_INTERVAL_SECONDS = 0.5
    _LOCK_PATH = Path(tempfile.gettempdir()) / "sequoia-x-akshare.lock"

    def __init__(self) -> None:
        self._lock_file: Any | None = None
        self._ak: Any | None = None
        self._last_request_at = 0.0
        self._consecutive_failures = 0
        self._tencent = _TencentQuoteSource()

    def __enter__(self) -> AkshareSession:
        self._lock_file = open(self._LOCK_PATH, "a+", encoding="utf-8")
        logger.info("已降级到 AKShare，等待独占访问锁")
        self._acquire_lock()
        import akshare as ak

        self._ak = ak
        return self

    def _acquire_lock(self) -> None:
        """带超时地抢独占锁。

        降级后的全市场同步会长时间持锁（2799 只约 47 分钟），若无超时，随后启动的
        日常任务会一直阻塞在锁上、在 crontab 里静默挂死。宁可放弃本轮 AKShare，
        也不让主流程无限等待。
        """
        if self._lock_file is None:
            return
        deadline = time.monotonic() + self.LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self._lock_file.close()
                    self._lock_file = None
                    raise RuntimeError(
                        f"等待 AKShare 独占访问锁超过 {self.LOCK_WAIT_SECONDS:.0f}s，"
                        "可能另有降级同步任务正在长时间占用。本次放弃 AKShare。"
                    ) from None
                time.sleep(self.LOCK_POLL_INTERVAL_SECONDS)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._ak = None
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def _throttle(self) -> None:
        """固定节流：距上次请求结束不足间隔就补足等待。"""
        delay = self.MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
        if delay > 0:
            time.sleep(delay)

    def _request(self, method: str, **kwargs: Any) -> Any:
        """带节流、短重试与连续失败熔断的 AKShare 调用。"""
        if self._ak is None:
            raise RuntimeError("AKShare 会话尚未初始化")
        if self._consecutive_failures >= self.CONSECUTIVE_FAILURE_LIMIT:
            raise RuntimeError(
                f"AKShare 已连续 {self._consecutive_failures} 次请求失败，判定上游正在限流，"
                "本轮熔断停止调用；请稍后再试，不要立即重跑。"
            )

        def invoke() -> Any:
            self._throttle()
            try:
                return getattr(self._ak, method)(**kwargs)
            finally:
                # 失败也打点：失败后密集重试正是把 IP 打进黑名单的行为。
                self._last_request_at = time.monotonic()

        try:
            result = _call_with_backoff(invoke, label=f"AKShare {method}")
        except Exception:
            self._consecutive_failures += 1
            raise
        self._consecutive_failures = 0
        return result

    @staticmethod
    def _symbol_from_bs_code(code: str) -> str:
        return code.split(".")[-1]

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        *,
        start_date: str,
        end_date: str,
        frequency: str,
        adjustflag: str,
    ) -> _RowsResult:
        if frequency != "d":
            raise ValueError("AKShare 降级源当前仅支持日线")
        adjust = {"1": "hfq", "2": "qfq", "3": ""}.get(adjustflag)
        if adjust is None:
            raise ValueError(f"不支持的复权标识: {adjustflag}")
        requested = fields.split(",")

        # 不复权路径优先走腾讯：东财 push2his 触发风控后会按 IP 直接断连，而这条路径
        # 只用于展示委托价、不落库，换源没有口径风险。腾讯缺 amount/turn（海龟策略要用
        # turn 算流通市值），字段不齐时直接交给东财。
        if adjustflag == "3" and set(requested) <= self._tencent.SUPPORTED_FIELDS:
            try:
                return self._tencent.query_history_k_data_plus(
                    code, fields, start_date=start_date, end_date=end_date,
                    frequency=frequency, adjustflag=adjustflag,
                )
            except Exception as exc:
                logger.warning(f"腾讯不复权日线不可用，回退东财: {exc}")

        try:
            df = self._request(
                "stock_zh_a_hist",
                symbol=self._symbol_from_bs_code(code),
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=adjust,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            if adjustflag == "1":
                # 后复权是唯一会写入 stock_daily 的口径，绝不能用腾讯 hfq 顶替：两者复权
                # 基准不同且无法换算，混存会静默污染策略数据（涨跌停判断依赖比值阈值）。
                raise RuntimeError(
                    f"后复权日线源（东方财富）不可用，腾讯后复权口径不同、不可替代: {exc}"
                ) from exc
            raise
        if df is None or df.empty:
            # 停牌、非交易日或代码不存在：与 Baostock 返回空结果集的行为对齐，
            # 不要因为「空表没有列名」而抛「缺少字段」这种误导性错误。
            return _RowsResult(requested, [])

        columns = {
            "date": "日期", "open": "开盘", "high": "最高", "low": "最低",
            "close": "收盘", "volume": "成交量", "amount": "成交额", "turn": "换手率",
        }
        missing = [
            name for name in requested if name not in columns or columns[name] not in df.columns
        ]
        if missing:
            raise RuntimeError(f"AKShare 日线缺少字段: {', '.join(missing)}")
        rows: list[list[str]] = []
        for _, item in df.iterrows():
            row: list[str] = []
            for name in requested:
                value = item[columns[name]]
                # 东方财富日线的成交量单位是手；本项目与 Baostock 一样统一存为股。
                if name == "volume":
                    value = float(value) * 100
                if name == "date":
                    value = str(value)[:10]
                row.append(str(value))
            rows.append(row)
        return _RowsResult(requested, rows)

    def query_stock_basic(self, *, code_name: str = "", code: str = "") -> _RowsResult:
        df = self._request("stock_info_a_code_name")
        if code:
            symbol = self._symbol_from_bs_code(code)
            df = df[df["code"].astype(str) == symbol]
        rows = []
        for _, item in df.iterrows():
            symbol = str(item["code"]).zfill(6)
            exchange = "sh" if symbol.startswith(("6", "9")) else "sz"
            rows.append([f"{exchange}.{symbol}", str(item["name"]), "", "", "1", "1"])
        return _RowsResult(["code", "code_name", "ipoDate", "outDate", "type", "status"], rows)


class MarketDataSession:
    """优先 Baostock，遇到登录、限额或接口错误自动降级 AKShare。"""

    def __init__(self) -> None:
        self._provider = "baostock"
        self._baostock: BaostockSession | None = None
        self._akshare: AkshareSession | None = None

    @property
    def provider(self) -> str:
        return self._provider

    def __enter__(self) -> MarketDataSession:
        try:
            self._baostock = BaostockSession()
            self._baostock.__enter__()
        except Exception as exc:
            logger.warning(f"Baostock 不可用，切换 AKShare: {exc}")
            self._baostock = None
            self._switch_to_akshare()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._baostock is not None:
            self._baostock.__exit__(exc_type, exc, traceback)
            self._baostock = None
        if self._akshare is not None:
            self._akshare.__exit__(exc_type, exc, traceback)
            self._akshare = None

    def _switch_to_akshare(self) -> None:
        if self._akshare is None:
            if self._baostock is not None:
                self._baostock.__exit__(None, None, None)
                self._baostock = None
            self._provider = "akshare"
            self._akshare = AkshareSession()
            self._akshare.__enter__()

    def _query(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._baostock is not None:
            try:
                result = getattr(self._baostock, method)(*args, **kwargs)
                if getattr(result, "error_code", "0") == "0":
                    return result
                logger.warning("Baostock 接口返回错误，切换 AKShare: %s", result.error_msg)
            except Exception as exc:
                logger.warning("Baostock 请求失败，切换 AKShare: %s", exc)
            self._switch_to_akshare()
        if self._akshare is None:
            raise RuntimeError("没有可用的行情数据源")
        return getattr(self._akshare, method)(*args, **kwargs)

    def query_history_k_data_plus(self, *args: Any, **kwargs: Any) -> Any:
        return self._query("query_history_k_data_plus", *args, **kwargs)

    def query_stock_basic(self, *args: Any, **kwargs: Any) -> Any:
        return self._query("query_stock_basic", *args, **kwargs)
