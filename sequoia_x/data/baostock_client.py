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

    def __enter__(self) -> "BaostockSession":
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
