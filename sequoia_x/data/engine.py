"""数据引擎模块：负责 SQLite 行情数据存储与 baostock 增量同步。"""

import sqlite3
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_daily (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    turnover REAL,
    UNIQUE (symbol, date)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_daily (symbol, date);
"""


class DataEngine:
    """行情数据引擎，负责 SQLite 存储和 baostock 数据同步。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path
        self.start_date: str = settings.start_date
        self._universe: list[str] | None = None
        self._init_db()

    def set_universe(self, symbols: list[str] | None) -> None:
        """设置本次运行的股票池；``None`` 表示使用本地全部股票。"""
        self._universe = list(dict.fromkeys(symbols)) if symbols is not None else None

    @property
    def is_universe_limited(self) -> bool:
        """当前运行是否限定了股票池。"""
        return self._universe is not None

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.commit()
        logger.info(f"数据库初始化完成：{self.db_path}")

    def _get_last_date(self, symbol: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM stock_daily WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                "SELECT * FROM stock_daily WHERE symbol = ? ORDER BY date",
                conn,
                params=(symbol,),
            )
        return df

    @staticmethod
    def _to_baostock_code(symbol: str) -> str:
        """将纯数字代码转为 baostock 格式：6/9开头 -> sh，其余 -> sz。"""
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        return f"{prefix}.{symbol}"

    # ── 数据同步 ──

    def sync_today_bulk(self, symbols: list[str] | None = None) -> int:
        """同步本地股票或指定股票池的增量日 K 数据（后复权）。

        Args:
            symbols: 指定时只同步该股票池；为 ``None`` 时同步本地全部股票。
                指定股票须已通过 ``backfill`` 写入本地数据库。
        """
        from datetime import date, timedelta
        from sequoia_x.data.baostock_client import MarketDataSession

        today_str = date.today().strftime("%Y-%m-%d")

        tasks = []
        with sqlite3.connect(self.db_path) as conn:
            if symbols is None:
                rows = conn.execute(
                    "SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol"
                ).fetchall()
            else:
                placeholders = ", ".join("?" for _ in symbols)
                rows = conn.execute(
                    f"SELECT symbol, MAX(date) FROM stock_daily "
                    f"WHERE symbol IN ({placeholders}) GROUP BY symbol",
                    symbols,
                ).fetchall()

        if not rows:
            scope = "指定股票池" if symbols is not None else "本地股票池"
            logger.warning(f"{scope}无股票数据，请先执行 --backfill")
            return 0

        for symbol, last_date in rows:
            if last_date and last_date >= today_str:
                continue
            start = today_str
            if last_date:
                start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            tasks.append((symbol, self._to_baostock_code(symbol), start, today_str))

        if not tasks:
            logger.info("所有股票已是最新，无需更新")
            return 0

        all_rows = []
        logger.info(f"需要更新 {len(tasks)} 只股票，按 Baostock 规则串行拉取...")
        try:
            with MarketDataSession() as bs:
                for symbol, bs_code, start, end in tasks:
                    rs = bs.query_history_k_data_plus(
                        bs_code, "date,open,high,low,close,volume,amount",
                        start_date=start, end_date=end, frequency="d", adjustflag="1",
                    )
                    if rs.error_code != "0":
                        logger.warning(f"[{symbol}] Baostock 查询失败: {rs.error_msg}")
                        continue
                    while rs.next():
                        all_rows.append([symbol] + rs.get_row_data())
        except Exception as exc:
            logger.error(f"行情同步失败（Baostock/AKShare）: {exc}")

        if not all_rows:
            logger.info("无新数据（可能非交易日）")
            return 0

        df = pd.DataFrame(all_rows, columns=["symbol", "date", "open", "high", "low", "close", "volume", "turnover"])
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["volume"] > 0]

        count = len(df)
        self._upsert_daily(df)

        logger.info(f"sync_today_bulk: 写入 {count} 条数据")
        return count

    def _upsert_daily(self, df: pd.DataFrame) -> None:
        """按 ``(symbol, date)`` 幂等写入日 K，避免部分同步覆盖其他股票。"""
        with sqlite3.connect(self.db_path) as conn:
            records = df[
                ["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]
            ].itertuples(index=False, name=None)
            conn.executemany(
                """
                INSERT INTO stock_daily (symbol, date, open, high, low, close, volume, turnover)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    turnover = excluded.turnover
                """,
                records,
            )
            conn.commit()

    def backfill(self, symbols: list[str]) -> None:
        """通过 baostock 批量回填历史日 K 线数据（后复权）。

        容错机制：
        - 单只股票失败自动重试 3 次，间隔递增（2s/4s/8s）
        - 全程使用一个受全局锁保护的串行连接，避免并发访问
        - 已入库的自动 skip，中断后可重跑续传
        """
        import time
        from datetime import date, timedelta

        from sequoia_x.data.baostock_client import MarketDataSession

        today_str = date.today().strftime("%Y-%m-%d")
        max_retries = 3
        success = 0
        skipped = 0
        failed = 0
        session: MarketDataSession | None = None
        try:
            session = MarketDataSession()
            bs = session.__enter__()
            total = len(symbols)
            for i, symbol in enumerate(symbols, start=1):
                progress = f"[{i}/{total}] [{symbol}]"
                last_date = self._get_last_date(symbol)
                if last_date and last_date >= today_str:
                    skipped += 1
                    logger.info(f"{progress} 跳过：数据已更新至 {last_date}")
                    continue

                start = last_date or self.start_date
                if last_date:
                    start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")

                bs_code = self._to_baostock_code(symbol)

                # 带重试的查询
                rows = []
                query_ok = False
                for attempt in range(max_retries):
                    try:
                        rs = bs.query_history_k_data_plus(
                            bs_code,
                            "date,open,high,low,close,volume,amount",
                            start_date=start,
                            end_date=today_str,
                            frequency="d",
                            adjustflag="1",  # 后复权
                        )

                        if rs.error_code != "0":
                            raise RuntimeError(rs.error_msg)

                        rows = []
                        while rs.next():
                            rows.append(rs.get_row_data())
                        query_ok = True
                        break

                    except Exception as exc:
                        if attempt < max_retries - 1:
                            wait = 2 ** (attempt + 1)
                            logger.warning(
                                f"[{symbol}] 第{attempt + 1}次失败: {exc}，{wait}s 后重试"
                            )
                            time.sleep(wait)
                        else:
                            logger.warning(f"[{symbol}] {max_retries}次重试均失败，跳过")

                if not query_ok:
                    failed += 1
                    logger.error(f"{progress} 失败：历史 K 线查询失败")
                    continue

                if not rows:
                    skipped += 1
                    logger.info(f"{progress} 跳过：{start} 至 {today_str} 无新增 K 线")
                    continue

                df = pd.DataFrame(rows, columns=rs.fields)
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                df = df[df["volume"] > 0]

                if df.empty:
                    skipped += 1
                    logger.info(f"{progress} 跳过：返回数据没有有效成交量")
                    continue

                df["symbol"] = symbol
                df = df.rename(columns={"amount": "turnover"})
                df = df[["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]]

                try:
                    with sqlite3.connect(self.db_path) as conn:
                        df.to_sql(
                            "stock_daily", conn, if_exists="append",
                            index=False, method="multi", chunksize=500,
                        )
                except sqlite3.IntegrityError:
                    pass

                success += 1
                end = str(df["date"].iloc[-1])
                logger.info(
                    f"{progress} 完成：写入 {len(df)} 条 K 线（{start} 至 {end}）；"
                    f"累计成功 {success}、跳过 {skipped}、失败 {failed}"
                )

        except Exception as exc:
            logger.error(f"历史回填失败（Baostock/AKShare）: {exc}")
        finally:
            if session is not None:
                session.__exit__(None, None, None)

        logger.info(f"回填完成 — 成功: {success} | 跳过: {skipped} | 失败: {failed}")

    # ── 股票列表 ──

    def get_all_symbols(self) -> list[str]:
        """通过 baostock 获取全市场 A 股代码列表。"""
        from sequoia_x.data.baostock_client import MarketDataSession

        try:
            with MarketDataSession() as bs:
                rs = bs.query_stock_basic(code_name="", code="")
                symbols = []
                while rs.next():
                    row = rs.get_row_data()
                    code = row[0]           # "sh.600000" or "sz.000001"
                    status = row[4]         # "1" = 上市
                    stock_type = row[5]     # "1" = 股票
                    if status == "1" and stock_type == "1":
                        symbols.append(code.split(".")[1])  # 提取纯数字代码
            logger.info(f"获取股票列表完成，共 {len(symbols)} 只")
            return symbols
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []

    def get_local_symbols(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            if self._universe is None:
                rows = conn.execute(
                    "SELECT DISTINCT symbol FROM stock_daily"
                ).fetchall()
            else:
                placeholders = ", ".join("?" for _ in self._universe)
                rows = conn.execute(
                    f"SELECT DISTINCT symbol FROM stock_daily WHERE symbol IN ({placeholders})",
                    self._universe,
                ).fetchall()
        return [row[0] for row in rows]
