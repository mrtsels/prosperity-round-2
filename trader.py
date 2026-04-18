"""
trader.py - Round 2 双股票量化交易策略（v16 ASH策略v2优化版）

策略：
1. INTARIAN_PEPPER_ROOT - 趋势动量策略（核心）
2. ASH_COATED_OSMIUM - 均值回归策略v2（布林带+动态仓位+分批止盈）

ASH v2核心改进:
- FV窗口: 15 → 8期
- 买卖阈值: ±4 → FV-1.5买, FV+2.0卖
- 布林带增强过滤
- 动态仓位管理
- 分批止盈(30%/40%/100%) + 移动止损

仓位限制：INTARIAN ≤ 80, ASH ≤ 30
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import math


# ============== 常量定义 ==============

PRODUCT_ASH = 'ASH_COATED_OSMIUM'
PRODUCT_INTARIAN = 'INTARIAN_PEPPER_ROOT'

POSITION_LIMIT_INTARIAN = 80
POSITION_LIMIT_ASH = 20

# INTARIAN 趋势动量参数
INTARIAN_LOOKBACK = 20
INTARIAN_STOP_LOSS_PCT = 0.015
INTARIAN_TRAILING_STOP_PCT = 0.025
INTARIAN_MA_SHORT = 5
INTARIAN_MA_LONG = 8
INTARIAN_ENTRY_CONSEC = 1
INTARIAN_ADD_CONSEC = 3
INTARIAN_FIRST_SIZE = 10
INTARIAN_ADD_SIZE = 10

# ASH 均值回归参数 (旧版，经过验证)
ASH_FV_WINDOW = 15
ASH_BUY_THRESH = 4
ASH_SELL_THRESH = 4


# ============== 工具函数 ==============

def get_mid_price(order_depth: OrderDepth) -> float:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0.0
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2


def get_best_bid_ask(order_depth: OrderDepth) -> tuple:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0, 0
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return best_bid, best_ask


def calc_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


# ============== 策略类 ==============

class MomentumStrategy:
    """INTARIAN_PEPPER_ROOT 趋势动量策略"""

    def __init__(self):
        self.position_limit = POSITION_LIMIT_INTARIAN
        self.price_history: Dict[str, List[float]] = {PRODUCT_INTARIAN: []}
        self.entry_price: Dict[str, float] = {}
        self.highest_price: Dict[str, float] = {}
        self.consecutive_uptrend = 0

    def update_history(self, product: str, price: float):
        if price <= 0:
            return
        if product not in self.price_history:
            self.price_history[product] = []
        self.price_history[product].append(price)
        if len(self.price_history[product]) > 100:
            self.price_history[product] = self.price_history[product][-100:]

    def signal(self, state: TradingState, product: str, position: int) -> List[Order]:
        orders = []
        od = state.order_depths.get(product)

        if od is None or not od.buy_orders or not od.sell_orders:
            return orders

        mid_price = get_mid_price(od)
        if mid_price <= 0:
            return orders

        self.update_history(product, mid_price)

        best_bid, best_ask = get_best_bid_ask(od)

        if product not in self.entry_price:
            self.entry_price[product] = mid_price
        if product not in self.highest_price:
            self.highest_price[product] = mid_price

        history = self.price_history.get(product, [])
        if len(history) < max(INTARIAN_MA_SHORT, INTARIAN_MA_LONG):
            return orders

        short_ma = sum(history[-INTARIAN_MA_SHORT:]) / INTARIAN_MA_SHORT
        long_ma = sum(history[-INTARIAN_MA_LONG:]) / INTARIAN_MA_LONG

        is_uptrend = short_ma > long_ma
        lookback_high = max(history[-INTARIAN_LOOKBACK:]) if len(history) >= INTARIAN_LOOKBACK else max(history)
        is_breakout = mid_price > lookback_high * 0.999

        if is_uptrend:
            self.consecutive_uptrend += 1
        else:
            self.consecutive_uptrend = 0

        available = self.position_limit - position

        # 入场
        if position == 0:
            if is_uptrend and self.consecutive_uptrend >= INTARIAN_ENTRY_CONSEC:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, INTARIAN_FIRST_SIZE)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    self.entry_price[product] = mid_price
                    self.highest_price[product] = mid_price

        # 加仓
        elif 0 < position < self.position_limit:
            if is_uptrend and is_breakout and self.consecutive_uptrend >= INTARIAN_ADD_CONSEC:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, INTARIAN_ADD_SIZE)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))

        # 持仓管理
        elif position > 0:
            if mid_price > self.highest_price[product]:
                self.highest_price[product] = mid_price

            entry = self.entry_price[product]
            peak = self.highest_price[product]

            if mid_price < entry * (1 - INTARIAN_STOP_LOSS_PCT):
                orders.append(Order(product, int(best_bid), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
                return orders

            trailing_stop = peak * (1 - INTARIAN_TRAILING_STOP_PCT)
            if mid_price < trailing_stop:
                orders.append(Order(product, int(best_bid), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
                return orders

        return orders


class ASHIntegerStrategy:
    """
    ASH_COATED_OSMIUM 整数均值回归策略

    严格使用整数价格，所有阈值均为绝对点数。
    无浮点计算、无布林带、无动态仓位。
    """

    # 参数常量
    FV_WINDOW = 10
    BUY_DELTA = 3
    SELL_DELTA = 3
    MAX_POSITION = 20
    BUY_SIZE = 8
    STOP_LOSS_DELTA = 12
    EXTREME_PRICE = 9900
    EXTREME_BUY_SIZE = 6
    EXTREME_SELL_PRICE = 9940
    EXTREME_STOP_PRICE = 9870

    def __init__(self):
        self.mid_price_history: List[int] = []
        self.entry_price: int = 0
        self.extreme_entry: int = 0
        self.stop_loss_triggered: bool = False

    def _calc_fv(self) -> int:
        if not self.mid_price_history:
            return 10000
        if len(self.mid_price_history) < self.FV_WINDOW:
            avg = sum(self.mid_price_history) / len(self.mid_price_history)
        else:
            recent = self.mid_price_history[-self.FV_WINDOW:]
            avg = sum(recent) / len(recent)
        return round(avg)

    def signal(self, state: TradingState, product: str, position: int) -> List[Order]:
        orders = []
        od = state.order_depths.get(product)

        if od is None or not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])

        if best_bid <= 0 or best_ask <= 0:
            return orders

        # 计算mid_price并更新历史
        mid_price = (best_bid + best_ask) // 2
        if mid_price > 0:
            self.mid_price_history.append(mid_price)
            if len(self.mid_price_history) > self.FV_WINDOW + 5:
                self.mid_price_history = self.mid_price_history[-(self.FV_WINDOW + 5):]

        fv = self._calc_fv()
        total_position = position

        # 止损检查
        if self.entry_price > 0:
            if best_bid <= self.entry_price - self.STOP_LOSS_DELTA:
                sell_qty = min(total_position, bid_vol)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    self.entry_price = 0
                    self.stop_loss_triggered = True
                    return orders

        if self.extreme_entry > 0:
            if best_bid <= self.EXTREME_STOP_PRICE:
                sell_qty = min(self.extreme_entry, bid_vol)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    self.extreme_entry = 0
                    return orders

        # 极端价格捕捉
        if not self.stop_loss_triggered and total_position < self.EXTREME_BUY_SIZE:
            if best_ask <= self.EXTREME_PRICE:
                buy_qty = min(self.EXTREME_BUY_SIZE - total_position, ask_vol)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    self.extreme_entry = best_ask
                    return orders

        # 极端仓止盈
        if self.extreme_entry > 0:
            if best_bid >= self.EXTREME_SELL_PRICE:
                orders.append(Order(product, best_bid, -self.extreme_entry))
                self.extreme_entry = 0
                return orders

        # 主交易逻辑
        if not self.stop_loss_triggered and total_position == 0:
            if best_ask <= fv - self.BUY_DELTA:
                buy_qty = min(self.BUY_SIZE, ask_vol)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    self.entry_price = best_ask
                    return orders

        # 卖出
        if total_position > 0:
            if best_bid >= fv + self.SELL_DELTA:
                sell_qty = min(total_position, bid_vol)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    self.entry_price = 0
                    self.extreme_entry = 0
                    return orders

        return orders


# ============== 主 Trader 类 ==============

class Trader:

    def __init__(self):
        self.bid_price = 20
        self.intarian_strategy = MomentumStrategy()
        self.ash_strategy = ASHIntegerStrategy()

    def bid(self) -> int:
        return self.bid_price

    def run(self, state: TradingState) -> tuple:
        result = {}

        position_intarian = state.position.get(PRODUCT_INTARIAN, 0)
        position_ash = state.position.get(PRODUCT_ASH, 0)

        # INTARIAN 趋势策略
        intarian_orders = self.intarian_strategy.signal(
            state, PRODUCT_INTARIAN, position_intarian
        )
        if intarian_orders:
            result[PRODUCT_INTARIAN] = intarian_orders

        # ASH 均值回归v2策略
        ash_orders = self.ash_strategy.signal(
            state, PRODUCT_ASH, position_ash
        )
        if ash_orders:
            result[PRODUCT_ASH] = ash_orders

        conversions = 0
        traderData = ""

        return result, conversions, traderData