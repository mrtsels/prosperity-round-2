"""
trader.py - Round 2 双股票量化交易策略（v15 优化INTARIAN+ASH动态FV版）

策略：
1. INTARIAN_PEPPER_ROOT - 趋势动量策略（核心）
2. ASH_COATED_OSMIUM - 动态FV均值回归策略

仓位限制：INTARIAN ≤ 80, ASH ≤ 30
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


# ============== 常量定义 ==============

PRODUCT_ASH = 'ASH_COATED_OSMIUM'
PRODUCT_INTARIAN = 'INTARIAN_PEPPER_ROOT'

POSITION_LIMIT_INTARIAN = 80
POSITION_LIMIT_ASH = 30

# INTARIAN 趋势动量参数 (优化版)
INTARIAN_LOOKBACK = 20
INTARIAN_STOP_LOSS_PCT = 0.015  # 1.5% 止损
INTARIAN_TRAILING_STOP_PCT = 0.025  # 2.5% 移动止损
INTARIAN_MA_SHORT = 5       # 短期均线
INTARIAN_MA_LONG = 8        # 长期均线
INTARIAN_ENTRY_CONSEC = 1   # 入场需连续上涨周期
INTARIAN_ADD_CONSEC = 3     # 加仓需连续上涨周期
INTARIAN_FIRST_SIZE = 10    # 首仓数量
INTARIAN_ADD_SIZE = 10      # 加仓数量

# ASH 动态FV均值回归参数
ASH_FV_WINDOW = 15      # 动态FV窗口
ASH_BUY_THRESH = 4     # 买入阈值：价格 < FV - 4
ASH_SELL_THRESH = 4    # 卖出阈值：价格 > FV + 4


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
        self.position_limit = POSITION_LIMIT_INTARIAN
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


class ASHDynamicFVStrategy:
    """
    ASH_COATED_OSMIUM 动态FV均值回归策略

    动态计算FV（滚动均值），价格低于FV时买入，高于FV时卖出
    """

    def __init__(self):
        self.position_limit = POSITION_LIMIT_ASH
        self.fv_window = ASH_FV_WINDOW
        self.buy_thresh = ASH_BUY_THRESH
        self.sell_thresh = ASH_SELL_THRESH
        self.price_history: List[float] = []
        self.entry_price: Dict[str, float] = {}

    def calc_fv(self) -> float:
        """计算动态Fair Value"""
        if len(self.price_history) < 2:
            return 10004  # 数据集均价
        if len(self.price_history) < self.fv_window:
            return sum(self.price_history) / len(self.price_history)
        return sum(self.price_history[-self.fv_window:]) / self.fv_window

    def signal(self,
               state: TradingState,
               product: str,
               position: int) -> List[Order]:
        orders = []
        od = state.order_depths.get(product)

        if od is None or not od.buy_orders or not od.sell_orders:
            return orders

        best_bid, best_ask = get_best_bid_ask(od)
        if best_bid <= 0 or best_ask <= 0:
            return orders

        mid_price = (best_bid + best_ask) / 2
        if mid_price <= 0:
            return orders

        # 更新价格历史
        self.price_history.append(mid_price)
        if len(self.price_history) > self.fv_window + 10:
            self.price_history = self.price_history[-(self.fv_window + 10):]

        # 计算动态FV
        fv = self.calc_fv()
        buy_line = fv - self.buy_thresh
        sell_line = fv + self.sell_thresh

        available = self.position_limit - position

        # 入场
        if position == 0:
            if mid_price <= buy_line:
                volume = min(available, 10)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    self.entry_price[product] = best_ask

        # 持仓管理
        elif position > 0:
            entry = self.entry_price.get(product, 0)

            # 止盈：价格达到卖出线
            if mid_price >= sell_line:
                orders.append(Order(product, int(best_bid), -position))
                self.entry_price[product] = 0
                return orders

            # 止损：价格跌破入场价99%
            if entry > 0 and mid_price < entry * 0.99:
                orders.append(Order(product, int(best_bid), -position))
                self.entry_price[product] = 0

        return orders


# ============== 主 Trader 类 ==============

class Trader:

    def __init__(self):
        self.bid_price = 20
        self.intarian_strategy = MomentumStrategy()
        self.ash_strategy = ASHDynamicFVStrategy()

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

        # ASH 动态FV均值回归策略
        ash_orders = self.ash_strategy.signal(
            state, PRODUCT_ASH, position_ash
        )
        if ash_orders:
            result[PRODUCT_ASH] = ash_orders

        conversions = 0
        traderData = ""

        return result, conversions, traderData