"""飞书通知模块：将选股结果通过 Webhook 推送至飞书群。"""

import json
from datetime import date

import requests

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class FeishuNotifier:
    """飞书 Webhook 推送器。

    所有策略统一推送到 Settings.feishu_webhook_url 配置的飞书机器人。
    """

    MAX_STOCKS_PER_STRATEGY = 3

    def __init__(self, settings: Settings, engine: DataEngine | None = None) -> None:
        """
        初始化 FeishuNotifier。

        Args:
            settings: Settings 实例，提供 Webhook URL 配置。
        """
        self.settings = settings
        self.engine = engine
        # ``stock_daily`` stores 后复权价 for strategy calculations.  A trade
        # plan, however, must quote the market's actual (unadjusted) price.
        # This cache is populated once per Feishu card build and is keyed by
        # (symbol, signal date).
        self._unadjusted_signal_prices: dict[tuple[str, str], tuple[float, float]] = {}

    @staticmethod
    def _to_xueqiu_code(code: str) -> str:
        """将纯数字代码转为雪球格式：6开头→SH，4/8开头→BJ，其余→SZ。"""
        if code.startswith("6"):
            return f"SH{code}"
        elif code.startswith(("4", "8")):
            return f"BJ{code}"
        return f"SZ{code}"

    @classmethod
    def select_symbols(cls, symbols: list[str]) -> list[str]:
        """保留策略评分后的全市场排名，只取前 3 名。"""
        return list(dict.fromkeys(symbols))[: cls.MAX_STOCKS_PER_STRATEGY]

    @staticmethod
    def _get_stock_names(symbols: list[str]) -> dict[str, str]:
        """通过 baostock 批量查询股票名称，返回 {code: name} 映射。"""
        from sequoia_x.data.baostock_client import BaostockSession

        mapping = {}
        try:
            with BaostockSession() as bs:
                for code in symbols:
                    prefix = "sh" if code.startswith(("6", "9")) else "sz"
                    rs = bs.query_stock_basic(code=f"{prefix}.{code}")
                    while rs.next():
                        row = rs.get_row_data()
                        mapping[code] = row[1]  # 第2个字段是股票名称
        except Exception as exc:
            logger.warning(f"Baostock 股票名称查询失败: {exc}")
        return mapping

    @staticmethod
    def _price(value: float) -> str:
        """将价格格式化为两位小数。"""
        return f"{value:.2f}"

    def _load_unadjusted_signal_prices(self, symbols: list[str]) -> None:
        """读取信号日不复权高低价，用于展示可实际下单的价格。

        策略库保留后复权价，避免除权除息破坏指标连续性；不能直接把该价格
        用作委托价。Baostock 的 ``adjustflag=3`` 是不复权口径。
        """
        if self.engine is None:
            return

        signal_dates: dict[str, str] = {}
        for symbol in symbols:
            df = self.engine.get_ohlcv(symbol)
            if not df.empty:
                signal_dates[symbol] = str(df.iloc[-1]["date"])[:10]

        if not signal_dates:
            return

        from sequoia_x.data.baostock_client import BaostockSession

        try:
            with BaostockSession() as bs:
                for symbol, signal_date in signal_dates.items():
                    code = self.engine._to_baostock_code(symbol)
                    rs = bs.query_history_k_data_plus(
                        code, "date,high,low", start_date=signal_date,
                        end_date=signal_date, frequency="d", adjustflag="3",
                    )
                    if rs.error_code != "0" or not rs.next():
                        logger.warning(f"[{symbol}] 未获取到信号日未复权价格: {rs.error_msg}")
                        continue
                    _, high, low = rs.get_row_data()
                    entry, stop = float(high), float(low)
                    if entry > stop > 0:
                        self._unadjusted_signal_prices[(symbol, signal_date)] = (entry, stop)
        except Exception as exc:
            # 行情补数失败时，通知仍应可以发送，只是不展示可能错误的价格。
            logger.warning(f"未复权交易计划价格获取失败: {exc}")

    def _trade_plan(self, symbol: str, strategy_name: str) -> tuple[str, str, str, str]:
        """为一个策略信号生成可执行的条件单计划。

        计划是规则化的风险提示，不是即时买卖指令：进场必须在下一交易日满足
        确认条件；止损和离场条件同时给出，避免只推送方向而没有退出纪律。
        """
        if strategy_name == "PrivatePlacementStrategy":
            return (
                "仅观察：先阅读定增用途、发行价/折价、认购方与锁定期；"
                "公告后不因消息本身直接买入。",
                "若定增方案终止、用途不及预期，或价格跌破公告日前低点，则放弃跟踪。",
                "基本面兑现后再评估；公告事件不设机械止盈。",
                "公告事件（非交易日信号）",
            )

        if self.engine is None:
            return (
                "下一交易日仅在放量突破信号日最高价时进场。",
                "收盘跌破信号日最低价即离场。",
                "达到 2 倍初始风险收益，或趋势条件失效时分批离场。",
                "未取得",
            )

        try:
            df = self.engine.get_ohlcv(symbol)
            if df.empty:
                raise ValueError("无本地K线")
            last = df.iloc[-1]
            signal_date = str(last["date"])[:10]
            prices = self._unadjusted_signal_prices.get((symbol, signal_date))
            if prices is None:
                raise ValueError("未取得信号日未复权价格")
            entry, stop = prices
            if entry <= stop or stop <= 0:
                raise ValueError("K线价格无效")
            target = entry + (entry - stop) * 2
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            logger.warning(f"[{symbol}] 无法生成精确交易计划：{exc}")
            return (
                "下一交易日仅在放量突破信号日最高价时进场。",
                "收盘跌破信号日最低价即离场。",
                "达到 2 倍初始风险收益，或趋势条件失效时分批离场。",
                "未取得",
            )

        entry_text = (
            f"下一交易日放量突破 {self._price(entry)} 后进场；"
            "未突破不追买。"
        )
        stop_text = f"收盘跌破信号日低点 {self._price(stop)}，次日离场。"
        exit_text = f"先看 {self._price(target)}（2R）；未到目标但趋势条件失效也离场。"

        if strategy_name == "LimitUpShakeoutStrategy":
            entry_text = f"下一交易日突破洗盘日高点 {self._price(entry)} 后进场；未突破不买。"
        elif strategy_name == "UptrendLimitDownStrategy":
            entry_text = f"不抄跌停：仅在下一交易日收复跌停日高点 {self._price(entry)} 后进场。"
        elif strategy_name == "HighTightFlagStrategy":
            entry_text = f"放量突破旗形整理高点 {self._price(entry)} 后进场；整理未破不买。"
        elif strategy_name == "RpsBreakoutStrategy":
            entry_text = f"RPS 保持强势且放量突破当日高点 {self._price(entry)} 后进场。"

        return entry_text, stop_text, exit_text, signal_date

    def _build_card(self, symbols: list[str], strategy_name: str) -> dict:
        symbols = self.select_symbols(symbols)
        self._load_unadjusted_signal_prices(symbols)
        today = date.today().strftime("%Y-%m-%d")
        names = self._get_stock_names(symbols)

        stock_elements: list[dict] = []
        for code in symbols:
            xq_code = self._to_xueqiu_code(code)
            name = names.get(code, xq_code)
            entry, stop, exit, signal_date = self._trade_plan(code, strategy_name)
            stock_elements.append(
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                            "content": (
                                f"**[{name}（{code}）](https://xueqiu.com/S/{xq_code})**\n"
                                f"**信号 K 线日期：** {signal_date}\n"
                                f"**进场：** {entry}\n"
                            f"**止损：** {stop}\n"
                            f"**离场：** {exit}"
                        ),
                    },
                }
            )

        elements: list[dict] = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        f"**日期：** {today}\n**策略：** {strategy_name}\n"
                        f"**推送数量：** {len(symbols)}（全市场最优前 3 只）"
                    ),
                },
            },
            {"tag": "hr"},
        ]
        elements.extend(stock_elements)

        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"📈 Sequoia-X 选股播报 | {strategy_name}",
                    },
                    "template": "blue",
                },
                "elements": elements,
            },
        }

    def send(
        self,
        symbols: list[str],
        strategy_name: str,
        webhook_key: str = "default",
    ) -> None:
        """
        将选股结果格式化为飞书卡片消息并 POST 至对应 Webhook。

        所有策略统一使用 FEISHU_WEBHOOK_URL。

        Args:
            symbols: 选股结果代码列表；仅推送前 3 只。
            strategy_name: 策略名称，用于卡片标题。
            webhook_key: 策略标识，仅用于日志标记。

        Raises:
            不抛出异常，HTTP 失败时记录 ERROR 日志。
        """
        url = self.settings.get_webhook_url(webhook_key)
        payload = self._build_card(symbols, strategy_name)

        try:
            resp = requests.post(
                url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            # 解析飞书真正的返回体
            resp_json = resp.json()

            # 飞书真正的成功标志是内部的 code == 0
            if resp.status_code != 200 or resp_json.get("code") != 0:
                logger.error(
                    f"飞书推送失败 [{webhook_key}] "
                    f"HTTP状态={resp.status_code} 飞书响应={resp.text}"
                )
            else:
                pushed = min(len(symbols), self.MAX_STOCKS_PER_STRATEGY)
                logger.info(f"飞书推送成功 [{webhook_key}]，共推送 {pushed} 只股票")

        except requests.RequestException as exc:
            logger.error(f"飞书推送请求异常 [{webhook_key}]：{exc}")
