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
from typing import Any
from zoneinfo import ZoneInfo

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class BaostockDailyLimitExceeded(RuntimeError):
    """在安全余量内达到 Baostock 自然日调用上限。"""


class BaostockSession:
    """独占、限速且受日配额保护的 Baostock 会话。"""

    # 官网上限为 50,000；预留 10% 余量，防止其他未计量客户端造成误判。
    DAILY_REQUEST_LIMIT = 45_000
    MIN_REQUEST_INTERVAL_SECONDS = 0.2
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


class AkshareSession:
    """AKShare 的串行节流会话，避免对其底层公开数据源施压。"""

    MIN_REQUEST_INTERVAL_SECONDS = 1.0
    _LOCK_PATH = Path(tempfile.gettempdir()) / "sequoia-x-akshare.lock"

    def __init__(self) -> None:
        self._lock_file: Any | None = None
        self._ak: Any | None = None
        self._last_request_at = 0.0

    def __enter__(self) -> AkshareSession:
        self._lock_file = open(self._LOCK_PATH, "a+", encoding="utf-8")
        logger.info("已降级到 AKShare，等待独占访问锁")
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        import akshare as ak

        self._ak = ak
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._ak = None
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def _request(self, method: str, **kwargs: Any) -> Any:
        if self._ak is None:
            raise RuntimeError("AKShare 会话尚未初始化")
        delay = self.MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
        if delay > 0:
            time.sleep(delay)
        result = getattr(self._ak, method)(**kwargs)
        self._last_request_at = time.monotonic()
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
        df = self._request(
            "stock_zh_a_hist",
            symbol=self._symbol_from_bs_code(code),
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust=adjust,
        )
        requested = fields.split(",")
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
