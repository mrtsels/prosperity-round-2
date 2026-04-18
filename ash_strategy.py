"""
ASH_COATED_OSMIUM 均值回归策略 (v2 优化版)

核心改进:
1. 动态FV窗口缩短至8期，对价格变化更敏感
2. 买卖阈值收窄(FV-1.5买, FV+2.0卖)，增加交易频率
3. 布林带增强过滤，避免噪音信号
4. 动态仓位管理，根据偏离度调整
5. 分批止盈+移动止损，保护利润
6. 资金再利用：ASH平仓后引导资金至INTARIAN

目标：从~100收益提升至800-1200+
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import math


# ============== 常量定义 ==============

PRODUCT_ASH = 'ASH_COATED_OSMIUM'
PRODUCT_INTARIAN = 'INTARIAN_PEPPER_ROOT'

POSITION_LIMIT_ASH = 30

# FV窗口
FV_WINDOW = 8                    # 动态FV窗口

# 买卖阈值
BUY_THRESH = 1.5                  # 买入: price <= FV - 1.5
SELL_THRESH = 2.0                # 卖出: price >= FV + 2.0
NEUTRAL_ZONE = 1.0               # 中性区: price在[FV-1.0, FV+1.0]时不动

# 布林带
BB_WINDOW = 8                    # 布林带窗口
BB_STD_MULT = 1.5                # 布林带标准差倍数

# 仓位管理
BASE_SIZE = 5                    # 基础开仓量
MAX_SIZE = 10                    # 最大单次开仓量
POSITION_LIMIT_INTARIAN = 80     # INTARIAN仓位上限


# ============== 工具函数 ==============

def get_mid_price(order_depth: OrderDepth) -> float:
    """获取中间价"""
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


def calc_std(values: List[float]) -> float:
    """计算标准差"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


# ============== ASH均值回归策略 ==============

class ASHMeanReversionStrategy:
    """
    ASH_COATED_OSMIUM 均值回归策略 v2

    改进点:
    - 动态FV(8期SMA)替代固定10000
    - 布林带过滤增强信号质量
    - 动态仓位管理
    - 分批止盈+移动止损
    """

    def __init__(self):
        self.position_limit = POSITION_LIMIT_ASH

        # 价格历史
        self.price_history: List[float] = []

        # 持仓状态
        self.entry_price: float = 0.0       # 开仓均价
        self.highest_price: float = 0.0     # 持仓期最高价
        self.position_opened: bool = False   # 是否已有持仓

        # 分批止盈状态
        self.tp30_triggered: bool = False   # 30%止盈已触发
        self.tp40_triggered: bool = False   # 40%止盈已触发

        # 追踪止损状态
        self.trail_enabled: bool = False     # 追踪止损是否启用
        self.trail_activated_price: float = 0.0  # 追踪止损激活时的价格

    def _update_price_history(self, price: float):
        """更新价格历史"""
        self.price_history.append(price)
        if len(self.price_history) > FV_WINDOW + 20:
            self.price_history = self.price_history[-(FV_WINDOW + 20):]

    def _calc_fv(self) -> float:
        """计算动态公平价值"""
        if len(self.price_history) < 2:
            return 10004  # fallback
        if len(self.price_history) < FV_WINDOW:
            return sum(self.price_history) / len(self.price_history)
        return sum(self.price_history[-FV_WINDOW:]) / FV_WINDOW

    def _calc_bollinger_bands(self) -> tuple:
        """
        计算布林带
        返回: (lower_band, middle_band, upper_band)
        """
        if len(self.price_history) < BB_WINDOW:
            # 历史不足，用FV作为中轨
            fv = self._calc_fv()
            return fv - 1.5, fv, fv + 1.5

        recent = self.price_history[-BB_WINDOW:]
        sma = sum(recent) / BB_WINDOW
        std = calc_std(recent)
        upper = sma + BB_STD_MULT * std
        lower = sma - BB_STD_MULT * std
        return lower, sma, upper

    def _calc_position_size(self, price: float, fv: float) -> int:
        """
        计算动态开仓量
        偏离度越大，仓位越大
        """
        deviation = abs(price - fv)
        extra = min(5, int((deviation - BUY_THRESH) * 2))
        size = BASE_SIZE + extra
        return min(MAX_SIZE, size)

    def _calc_take_profit(self, position: int, current_price: float) -> List[Order]:
        """
        分批止盈逻辑
        浮盈>=1.0%: 卖出30%
        浮盈>=1.5%: 卖出40%
        浮盈>=2.0%: 卖出剩余
        """
        orders = []
        if self.entry_price <= 0:
            return orders

        profit_pct = (current_price - self.entry_price) / self.entry_price

        # 计算各档次应平仓量
        remaining = position
        to_sell = 0

        if profit_pct >= 0.01 and not self.tp30_triggered:
            to_sell = int(position * 0.30)
            self.tp30_triggered = True

        if profit_pct >= 0.015 and not self.tp40_triggered:
            to_sell += int(position * 0.40)
            self.tp40_triggered = True

        if profit_pct >= 0.02:
            to_sell = remaining  # 全部平仓

        if to_sell > 0:
            orders.append(Order(PRODUCT_ASH, int(current_price), -to_sell))

        return orders

    def _calc_trailing_stop(self, position: int, current_price: float) -> List[Order]:
        """
        移动止损逻辑
        当浮盈超过0.8%时启用追踪止损
        回撤0.5%时触发止损
        """
        orders = []
        if self.entry_price <= 0 or position <= 0:
            return orders

        profit_pct = (current_price - self.entry_price) / self.entry_price

        # 启用追踪止损
        if profit_pct >= 0.008 and not self.trail_enabled:
            self.trail_enabled = True
            self.trail_activated_price = current_price

        # 执行追踪止损
        if self.trail_enabled and current_price < self.trail_activated_price * (1 - 0.005):
            orders.append(Order(PRODUCT_ASH, int(current_price), -position))
            # 重置状态
            self.trail_enabled = False
            self.trail_activated_price = 0.0

        return orders

    def _reset_state(self):
        """重置持仓状态"""
        self.entry_price = 0.0
        self.highest_price = 0.0
        self.position_opened = False
        self.tp30_triggered = False
        self.tp40_triggered = False
        self.trail_enabled = False
        self.trail_activated_price = 0.0

    def signal(self, state: TradingState, product: str, position: int) -> List[Order]:
        """
        生成交易信号

        Args:
            state: TradingState对象
            product: 产品名称
            position: 当前持仓

        Returns:
            List[Order]: 订单列表
        """
        orders = []
        od = state.order_depths.get(product)

        if od is None or not od.buy_orders or not od.sell_orders:
            return orders

        best_bid, best_ask = get_best_bid_ask(od)
        if best_bid <= 0 or best_ask <= 0:
            return orders

        mid_price = get_mid_price(od)
        if mid_price <= 0:
            return orders

        # 更新价格历史
        self._update_price_history(mid_price)

        # 计算FV和布林带
        fv = self._calc_fv()
        bb_lower, bb_middle, bb_upper = self._calc_bollinger_bands()

        buy_line = fv - BUY_THRESH
        sell_line = fv + SELL_THRESH
        neutral_low = fv - NEUTRAL_ZONE
        neutral_high = fv + NEUTRAL_ZONE

        available = self.position_limit - position

        # ========== 入场逻辑 ==========
        if position == 0:
            # 买入信号：价格<=FV-1.5 且 价格<=布林带下轨
            if mid_price <= buy_line and mid_price <= bb_lower:
                size = self._calc_position_size(mid_price, fv)
                size = min(size, available)
                if size > 0:
                    orders.append(Order(product, int(best_ask), size))
                    self.entry_price = best_ask
                    self.highest_price = mid_price
                    self.position_opened = True

        # ========== 持仓管理 ==========
        elif position > 0:
            # 更新最高价
            if mid_price > self.highest_price:
                self.highest_price = mid_price

            # 卖出信号：价格>=FV+2.0 且 价格>=布林带上轨
            if mid_price >= sell_line and mid_price >= bb_upper:
                orders.append(Order(product, int(best_bid), -position))
                self._reset_state()
                return orders

            # 分批止盈
            tp_orders = self._calc_take_profit(position, mid_price)
            orders.extend(tp_orders)

            # 移动止损
            if not orders:  # 如果没有止盈，才检查止损
                trail_orders = self._calc_trailing_stop(position, mid_price)
                orders.extend(trail_orders)

            # 如果所有订单都平仓了，重置状态
            if orders and sum(abs(o.quantity) for o in orders) >= position:
                self._reset_state()

        return orders


# ============== 资金再利用逻辑 (主Trader用) ==============

class CapitalRecyclingManager:
    """
    资金再利用管理器

    思路说明：
    当ASH策略平仓后，释放出的资金可以引导至INTARIAN策略，
    前提是INTARIAN当时处于明确上升趋势且未满仓。

    实现方式：
    1. 监控ASH持仓变化（通过position差异检测平仓）
    2. 检查INTARIAN状态（均线多头排列、突破）
    3. 若满足条件，允许INTARIAN下次信号时加大仓位

    注意：这只是一个思路示例，实际需要在Trader主循环中配合实现
    """

    def __init__(self):
        self.last_ash_position = 0
        self.ash_just_closed = False    # ASH刚平仓标志
        self.intarian_can_boost = False  # INTARIAN可加大仓位

    def update(self, current_ash_position: int, intarian_trend_up: bool,
               intarian_breakout: bool, intarian_position: int):
        """
        每周期调用，更新资金再利用状态
        """
        # 检测ASH平仓
        if self.last_ash_position > 0 and current_ash_position == 0:
            self.ash_just_closed = True

            # 若INTARIAN趋势良好，可加大仓位
            if intarian_trend_up and intarian_breakout and intarian_position < POSITION_LIMIT_INTARIAN:
                self.intarian_can_boost = True

        self.last_ash_position = current_ash_position

        # INTARIAN加仓后，重置标志
        if self.intarian_can_boost and intarian_position >= POSITION_LIMIT_INTARIAN - 10:
            self.intarian_can_boost = False

    def get_bonus_allocation(self) -> int:
        """获取额外可分配的仓位"""
        if self.intarian_can_boost:
            return 10  # 额外10手额度
        return 0


# ============== 使用示例 ==============

def example_trader_usage():
    """
    展示如何在主Trader中集成ASH策略
    """

    # 创建策略实例
    ash_strategy = ASHMeanReversionStrategy()
    capital_recycler = CapitalRecyclingManager()

    def run(self, state: TradingState) -> tuple:
        result = {}

        # INTARIAN处理
        position_intarian = state.position.get(PRODUCT_INTARIAN, 0)
        intarian_orders = self.intarian_strategy.signal(
            state, PRODUCT_INTARIAN, position_intarian
        )
        if intarian_orders:
            result[PRODUCT_INTARIAN] = intarian_orders

        # ASH处理
        position_ash = state.position.get(PRODUCT_ASH, 0)

        # 更新资金再利用状态
        intarian_hist = self.intarian_strategy.price_history.get(PRODUCT_INTARIAN, [])
        intarian_trend_up = False
        intarian_breakout = False
        if len(intarian_hist) >= 10:
            ma5 = sum(intarian_hist[-5:]) / 5
            ma10 = sum(intarian_hist[-10:]) / 10
            intarian_trend_up = ma5 > ma10
            lookback_high = max(intarian_hist[-5:]) if len(intarian_hist) >= 5 else intarian_hist[-1]
            intarian_breakout = (state.order_depths[PRODUCT_INTARIAN].buy_orders and
                                (max(state.order_depths[PRODUCT_INTARIAN].buy_orders.keys()) >
                                 lookback_high * 0.999))

        capital_recycler.update(position_ash, intarian_trend_up,
                               intarian_breakout, position_intarian)

        # ASH信号
        ash_orders = ash_strategy.signal(state, PRODUCT_ASH, position_ash)
        if ash_orders:
            result[PRODUCT_ASH] = ash_orders

        return result, 0, ""

    return run


if __name__ == "__main__":
    # 简单测试
    print("ASHMeanReversionStrategy v2 已定义")
    print(f"参数: FV窗口={FV_WINDOW}, 买入阈值={BUY_THRESH}, 卖出阈值={SELL_THRESH}")
    print(f"布林带: 窗口={BB_WINDOW}, 倍数={BB_STD_MULT}")
    print(f"仓位: 基础={BASE_SIZE}, 最大={MAX_SIZE}")
    print()
    print("预期效果:")
    print("- 交易频率: 5-8次买卖对")
    print("- 单次收益: 15-30点")
    print("- 目标总收益: 800-1200+")