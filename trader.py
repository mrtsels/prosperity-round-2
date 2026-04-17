"""
trader.py - Round 2 双股票量化交易策略（v14 INTARIAN+ASH组合版）

策略：
1. INTARIAN_PEPPER_ROOT - 趋势动量策略（核心）
2. ASH_COATED_OSMIUM - 均值回归策略（辅助）

仓位限制：INTARIAN ≤ 80，ASH ≤ 30
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


# ============== 常量定义 ==============

PRODUCT_ASH = 'ASH_COATED_OSMIUM'
PRODUCT_INTARIAN = 'INTARIAN_PEPPER_ROOT'

POSITION_LIMIT = 80

# INTARIAN 趋势动量参数
INTARIAN_LOOKBACK = 20
INTARIAN_STOP_LOSS_PCT = 0.015  # 1.5% 止损
INTARIAN_TRAILING_STOP_PCT = 0.025  # 2.5% 移动止损
INTARIAN_MAX_POSITION = 80

# ASH 波段策略参数
ASH_MAX_POSITION = 30
ASH_CENTER = 10000
ASH_BUY_ZONE = 5    # 价格 < center - 5 时买入
ASH_SELL_ZONE = 5   # 价格 > center + 5 时卖出
ASH_STOP_OFFSET = 10  # 止损偏移


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


# ============== 策略类 ==============

class MomentumStrategy:
    """
    INTARIAN_PEPPER_ROOT 趋势动量策略
    """

    def __init__(self):
        self.position_limit = INTARIAN_MAX_POSITION
        self.price_history: Dict[str, List[float]] = {
            PRODUCT_INTARIAN: []
        }
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

    def signal(self,
               state: TradingState,
               product: str,
               position: int) -> List[Order]:
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
        if len(history) < 5:
            return orders

        short_ma = sum(history[-5:]) / 5
        long_ma = sum(history[-10:]) / 10 if len(history) >= 10 else short_ma

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
            if is_uptrend and self.consecutive_uptrend >= 3:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, 20)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    self.entry_price[product] = mid_price
                    self.highest_price[product] = mid_price

        # 加仓
        elif 0 < position < self.position_limit:
            if is_uptrend and is_breakout and self.consecutive_uptrend >= 5:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, 20)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))

        # 持仓管理
        elif position > 0:
            if mid_price > self.highest_price[product]:
                self.highest_price[product] = mid_price

            entry = self.entry_price[product]
            peak = self.highest_price[product]

            if mid_price < entry * (1 - INTARIAN_STOP_LOSS_PCT):
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
                return orders

            trailing_stop = peak * (1 - INTARIAN_TRAILING_STOP_PCT)
            if mid_price < trailing_stop:
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
                return orders

        return orders


class ASHSwingStrategy:
    """
    ASH_COATED_OSMIUM 波段策略
    价格围绕中心线10000上下波动，低买高卖
    """

    def __init__(self):
        self.position_limit = ASH_MAX_POSITION
        self.center = ASH_CENTER
        self.price_history: Dict[str, List[float]] = {
            PRODUCT_ASH: []
        }
        self.entry_price: Dict[str, float] = {}

    def update_history(self, product: str, price: float):
        if price <= 0:
            return
        if product not in self.price_history:
            self.price_history[product] = []
        self.price_history[product].append(price)
        if len(self.price_history[product]) > 100:
            self.price_history[product] = self.price_history[product][-100:]

    def signal(self,
               state: TradingState,
               product: str,
               position: int) -> List[Order]:
        orders = []
        od = state.order_depths.get(product)

        if od is None or not od.buy_orders or not od.sell_orders:
            return orders

        mid_price = get_mid_price(od)
        if mid_price <= 0:
            return orders

        self.update_history(product, mid_price)

        # 动态计算中心线（20日均线）
        history = self.price_history.get(product, [])
        if len(history) >= 20:
            center_line = sum(history[-20:]) / 20
        else:
            center_line = self.center

        buy_line = center_line - ASH_BUY_ZONE
        sell_line = center_line + ASH_SELL_ZONE
        stop_line = center_line - ASH_STOP_OFFSET

        available = self.position_limit - position

        # 入场：价格低于 buy_line
        if position == 0:
            if mid_price <= buy_line:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, 20)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    self.entry_price[product] = mid_price

        # 持仓管理
        elif position > 0:
            entry = self.entry_price.get(product, mid_price)

            # 止盈：价格高于 sell_line
            if mid_price >= sell_line:
                best_bid = max(od.buy_orders.keys())
                orders.append(Order(product, int(best_bid), -position))
                self.entry_price[product] = 0
                return orders

            # 止损：价格跌破 stop_line
            if mid_price <= stop_line:
                best_bid = max(od.buy_orders.keys())
                orders.append(Order(product, int(best_bid), -position))
                self.entry_price[product] = 0

        return orders


# ============== 主 Trader 类 ==============

class Trader:

    def __init__(self):
        self.bid_price = 20
        self.intarian_strategy = MomentumStrategy()
        self.ash_strategy = ASHSwingStrategy()

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

        # ASH 波段策略
        ash_orders = self.ash_strategy.signal(
            state, PRODUCT_ASH, position_ash
        )
        if ash_orders:
            result[PRODUCT_ASH] = ash_orders

        conversions = 0
        traderData = ""

        return result, conversions, traderData