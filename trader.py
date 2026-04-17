"""
trader.py - Round 2 双股票量化交易策略

策略：
1. INTARIAN_PEPPER_ROOT - 趋势动量策略（核心）
2. ASH_COATED_OSMIUM - 均值回归策略（辅助）

仓位限制：每只股票 ≤ 80
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import math


# ============== 常量定义 ==============

PRODUCT_ASH = 'ASH_COATED_OSMIUM'
PRODUCT_INTARIAN = 'INTARIAN_PEPPER_ROOT'

POSITION_LIMIT = 80

# ASH 均值回归参数
ASH_CENTER = 10_000
ASH_BUY_THRESHOLD = 9_985
ASH_SELL_THRESHOLD = 10_015
ASH_STOP_LOSS_PCT = 0.002  # 0.2%

# INTARIAN 动量参数
INTARIAN_LOOKBACK = 20
INTARIAN_STOP_LOSS_PCT = 0.02  # 2%
INTARIAN_TAKE_PROFIT_PCT = 0.03  # 3%

# OBI 过滤参数
OBI_THRESHOLD = 0.15


# ============== 工具函数 ==============

def get_mid_price(order_depth: OrderDepth) -> float:
    """计算订单簿中间价"""
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0.0
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2


def get_best_bid_ask(order_depth: OrderDepth) -> tuple:
    """获取最优买卖价"""
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0, 0
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return best_bid, best_ask


def compute_obi(order_depth: OrderDepth) -> float:
    """计算订单簿不平衡度"""
    total_bid = sum(order_depth.buy_orders.values())
    total_ask = sum(abs(v) for v in order_depth.sell_orders.values())

    if total_bid + total_ask == 0:
        return 0.0
    return (total_bid - total_ask) / (total_bid + total_ask)


def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 10) -> float:
    """计算ATR（Average True Range）"""
    if len(highs) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, min(period + 1, len(highs))):
        tr = max(
            highs[-i] - lows[-i],
            abs(highs[-i] - closes[-i-1]),
            abs(lows[-i] - closes[-i-1])
        )
        true_ranges.append(tr)

    return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0


# ============== 策略类 ==============

class MomentumStrategy:
    """
    INTARIAN_PEPPER_ROOT 趋势动量策略

    逻辑：
    - 趋势跟踪：价格突破时买入
    - 移动止损：保护利润
    - 仓位管理：不超过限制
    """

    def __init__(self):
        self.position_limit = POSITION_LIMIT
        self.price_history: Dict[str, List[float]] = {
            PRODUCT_INTARIAN: []
        }
        self.entry_price: Dict[str, float] = {}
        self.highest_price: Dict[str, float] = {}

    def update_history(self, product: str, price: float):
        """更新价格历史"""
        if product not in self.price_history:
            self.price_history[product] = []
        self.price_history[product].append(price)
        # 保留足够的历史数据
        if len(self.price_history[product]) > 100:
            self.price_history[product] = self.price_history[product][-100:]

    def signal(self,
               state: TradingState,
               product: str,
               position: int) -> List[Order]:
        """
        生成交易信号

        Args:
            state: TradingState
            product: 产品名
            position: 当前持仓

        Returns:
            订单列表
        """
        orders = []
        od = state.order_depths.get(product)

        if od is None or not od.buy_orders or not od.sell_orders:
            return orders

        mid_price = get_mid_price(od)
        if mid_price <= 0:
            return orders

        self.update_history(product, mid_price)

        # 初始化
        if product not in self.entry_price:
            self.entry_price[product] = mid_price
        if product not in self.highest_price:
            self.highest_price[product] = mid_price

        # 趋势判断：基于最近N个价格的方向
        history = self.price_history.get(product, [])
        if len(history) < 5:
            return orders

        # 计算短期均线
        short_ma = sum(history[-5:]) / 5
        long_ma = sum(history[-10:]) / 10 if len(history) >= 10 else short_ma

        # 趋势信号：短期均线 > 长期均线
        is_uptrend = short_ma > long_ma

        # 突破信号：价格创N日新高
        lookback_high = max(history[-INTARIAN_LOOKBACK:]) if len(history) >= INTARIAN_LOOKBACK else max(history)
        is_breakout = mid_price > lookback_high * 0.998  # 接近新高

        # 可用仓位
        available = self.position_limit - position

        # 入场逻辑：上升趋势 + 突破
        if position == 0:
            if is_uptrend and is_breakout:
                # 买入信号
                best_ask = min(od.sell_orders.keys())
                volume = min(available, 10)  # 每次最多买10手
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    self.entry_price[product] = mid_price
                    self.highest_price[product] = mid_price

        # 持仓管理
        elif position > 0:
            # 更新最高价
            if mid_price > self.highest_price[product]:
                self.highest_price[product] = mid_price

            entry = self.entry_price[product]

            # 止损检查
            if mid_price < entry * (1 - INTARIAN_STOP_LOSS_PCT):
                # 止损卖出
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0
                return orders

            # 止盈检查
            if mid_price > entry * (1 + INTARIAN_TAKE_PROFIT_PCT):
                # 止盈卖出一半
                sell_qty = min(position // 2, 10)
                if sell_qty > 0:
                    orders.append(Order(product, int(mid_price), -sell_qty))

            # 移动止损：价格从高点回撤超过2%则退出
            trailing_stop = self.highest_price[product] * (1 - 0.015)
            if mid_price < trailing_stop:
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0

        return orders


class MeanReversionStrategy:
    """
    ASH_COATED_OSMIUM 均值回归策略

    逻辑：
    - 价格偏离均值时买入
    - OBI过滤确认方向
    - 快速止损
    """

    def __init__(self):
        self.position_limit = POSITION_LIMIT
        self.price_history: Dict[str, List[float]] = {
            PRODUCT_ASH: []
        }
        self.entry_price: Dict[str, float] = {}

    def update_history(self, product: str, price: float):
        """更新价格历史"""
        if product not in self.price_history:
            self.price_history[product] = []
        self.price_history[product].append(price)
        if len(self.price_history[product]) > 100:
            self.price_history[product] = self.price_history[product][-100:]

    def signal(self,
               state: TradingState,
               product: str,
               position: int) -> List[Order]:
        """
        生成交易信号

        Args:
            state: TradingState
            product: 产品名
            position: 当前持仓

        Returns:
            订单列表
        """
        orders = []
        od = state.order_depths.get(product)

        if od is None or not od.buy_orders or not od.sell_orders:
            return orders

        mid_price = get_mid_price(od)
        if mid_price <= 0:
            return orders

        self.update_history(product, mid_price)

        # OBI过滤
        obi = compute_obi(od)

        # 计算动态中心线（基于最近均值）
        history = self.price_history.get(product, [])
        if len(history) >= 10:
            center = sum(history[-10:]) / 10
        else:
            center = ASH_CENTER

        # 可用仓位
        available = self.position_limit - position

        # 入场逻辑：价格低于阈值 + OBI偏多
        if position == 0:
            if mid_price <= ASH_BUY_THRESHOLD and obi > OBI_THRESHOLD:
                # 买入信号
                best_ask = min(od.sell_orders.keys())
                volume = min(available, 15)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    self.entry_price[product] = mid_price

        # 持仓管理
        elif position > 0:
            entry = self.entry_price.get(product, mid_price)

            # 止损：价格没有回归则止损
            if mid_price > center * 1.001:  # 价格继续偏离
                # 全部卖出
                orders.append(Order(product, int(mid_price), -position))
                return orders

            # 止盈：价格回到中心线附近
            if mid_price >= center * 0.9995:
                # 全部卖出
                orders.append(Order(product, int(mid_price), -position))
                return orders

        return orders


# ============== 主 Trader 类 ==============

class Trader:

    def __init__(self):
        """初始化交易器"""
        # MAF竞拍价格
        self.bid_price = 20

        # 策略实例
        self.intarian_strategy = MomentumStrategy()
        self.ash_strategy = MeanReversionStrategy()

    def bid(self) -> int:
        """
        Market Access Fee 竞拍价格

        博弈论策略：只需进入前50%，bid过高浪费利润
        """
        return self.bid_price

    def run(self, state: TradingState) -> tuple:
        """
        主交易逻辑

        Args:
            state: TradingState

        Returns:
            (orders, conversions, traderData)
        """
        # 初始化结果
        result = {}

        # 获取当前持仓
        position_ash = state.position.get(PRODUCT_ASH, 0)
        position_intarian = state.position.get(PRODUCT_INTARIAN, 0)

        # ============== INTARIAN 动量策略 ==============
        intarian_orders = self.intarian_strategy.signal(
            state, PRODUCT_INTARIAN, position_intarian
        )
        if intarian_orders:
            result[PRODUCT_INTARIAN] = intarian_orders

        # ============== ASH 均值回归策略 ==============
        ash_orders = self.ash_strategy.signal(
            state, PRODUCT_ASH, position_ash
        )
        if ash_orders:
            result[PRODUCT_ASH] = ash_orders

        # ============== 返回结果 ==============
        conversions = 0
        traderData = ""  # 可以序列化状态

        return result, conversions, traderData
