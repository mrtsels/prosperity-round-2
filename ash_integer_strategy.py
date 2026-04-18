"""
ASH_COATED_OSMIUM 均值回归策略 (整数版)

严格适配整数价格环境，所有阈值均为绝对点数。
无浮点计算、无布林带、无动态仓位。

策略模块：
1. 主交易逻辑：FV偏差±6/7点触发
2. 极端价格捕捉：价格<=9900时买入，>=9940或<=9870时平仓
3. 固定止损：偏离入场价12点时止损

持仓上限：20手
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


# ============== 常量定义 ==============

PRODUCT_ASH = 'ASH_COATED_OSMIUM'

# 主交易参数
FV_WINDOW = 10              # FV窗口期数
BUY_DELTA = 6              # 买入触发: best_ask <= FV - 6
SELL_DELTA = 7             # 卖出触发: best_bid >= FV + 7
MAX_POSITION = 20           # 持仓上限
BUY_SIZE = 8               # 买入数量
STOP_LOSS_DELTA = 12       # 止损: 偏离入场价12点

# 极端价格捕捉
EXTREME_PRICE = 9900       # 极端价格阈值
EXTREME_BUY_SIZE = 6       # 极端买入数量
EXTREME_SELL_PRICE = 9940  # 极端仓位止盈价
EXTREME_STOP_PRICE = 9870  # 极端仓位止损价


# ============== 工具函数 ==============

def get_order_depth(state: TradingState, product: str) -> OrderDepth:
    """获取订单簿，不存在时返回空OrderDepth"""
    return state.order_depths.get(product, OrderDepth())


def get_best_bid_ask(order_depth: OrderDepth) -> tuple:
    """
    获取最优买卖价和挂单量
    返回: (best_bid, bid_vol, best_ask, ask_vol)
    """
    if not order_depth.buy_orders:
        return 0, 0, 0, 0
    if not order_depth.sell_orders:
        return 0, 0, 0, 0

    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    bid_vol = order_depth.buy_orders[best_bid]
    ask_vol = abs(order_depth.sell_orders[best_ask])

    return best_bid, bid_vol, best_ask, ask_vol


def calc_mid_price(best_bid: int, best_ask: int) -> int:
    """计算中间价（整数）"""
    if best_bid <= 0 or best_ask <= 0:
        return 0
    return (best_bid + best_ask) // 2


# ============== ASH整数均值回归策略 ==============

class ASHIntegerStrategy:
    """
    ASH整数均值回归策略

    严格使用整数价格，所有阈值均为绝对点数。

    策略逻辑：
    1. 主交易：FV偏差±6/7点触发买卖
    2. 极端捕捉：价格<=9900时买入极端机会仓
    3. 止损：偏离入场价12点止损，止损后当日不再开仓
    """

    def __init__(self):
        self.max_position = MAX_POSITION

        # 价格历史（存mid_price，整数）
        self.mid_price_history: List[int] = []

        # 持仓状态
        self.position: int = 0              # 当前持仓
        self.entry_price: int = 0          # 主仓入场价
        self.extreme_position: int = 0     # 极端仓持仓
        self.extreme_entry: int = 0        # 极端仓入场价

        # 止损标志
        self.stop_loss_triggered: bool = False  # 止损后当日不再开仓

    def _update_history(self, mid_price: int):
        """更新价格历史"""
        if mid_price <= 0:
            return
        self.mid_price_history.append(mid_price)
        # 保持最近N期历史
        if len(self.mid_price_history) > FV_WINDOW + 5:
            self.mid_price_history = self.mid_price_history[-(FV_WINDOW + 5):]

    def _calc_fv(self) -> int:
        """
        计算公平价值（整数）
        FV = 四舍五入取整(最近10期mid_price均值)
        """
        if not self.mid_price_history:
            return 10000  # 默认值

        if len(self.mid_price_history) < FV_WINDOW:
            avg = sum(self.mid_price_history) / len(self.mid_price_history)
        else:
            recent = self.mid_price_history[-FV_WINDOW:]
            avg = sum(recent) / len(recent)

        # 四舍五入取整
        return round(avg)

    def _reset_daily_flags(self):
        """每日重置标志（新一天开始时调用）"""
        self.stop_loss_triggered = False


# ============== 信号生成 ==============

    def signal(self, state: TradingState, product: str, position: int) -> List[Order]:
        """
        生成交易信号

        Args:
            state: TradingState对象
            product: 产品名称
            position: 当前持仓（来自state）

        Returns:
            List[Order]: 订单列表
        """
        orders = []

        # 获取订单簿数据
        od = get_order_depth(state, product)
        best_bid, bid_vol, best_ask, ask_vol = get_best_bid_ask(od)

        # 有效性检查
        if best_bid <= 0 or best_ask <= 0:
            return orders

        # 计算中间价并更新历史
        mid_price = calc_mid_price(best_bid, best_ask)
        if mid_price > 0:
            self._update_history(mid_price)

        # 更新外部传入的持仓
        self.position = position
        total_position = position

        # 计算FV
        fv = self._calc_fv()

        # ========== 优先级1：止损检查 ==========
        # 检查主仓止损
        if self.entry_price > 0:
            if best_bid <= self.entry_price - STOP_LOSS_DELTA:
                # 止损卖出
                sell_qty = min(total_position, bid_vol)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    self.entry_price = 0
                    self.stop_loss_triggered = True
                    return orders

        # 检查极端仓止损
        if self.extreme_entry > 0:
            if best_bid <= EXTREME_STOP_PRICE:
                # 极端仓止损
                sell_qty = min(self.extreme_position, bid_vol)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    self.extreme_position = 0
                    self.extreme_entry = 0
                    return orders

        # ========== 优先级2：极端价格捕捉 ==========
        # 条件：无持仓或总仓位<6，且价格<=9900，且未触发止损
        if not self.stop_loss_triggered:
            if total_position < EXTREME_BUY_SIZE:
                if best_ask <= EXTREME_PRICE:
                    buy_qty = min(EXTREME_BUY_SIZE - total_position, ask_vol)
                    if buy_qty > 0:
                        orders.append(Order(product, best_ask, buy_qty))
                        self.extreme_position += buy_qty
                        self.extreme_entry = best_ask
                        # 注意：这里不return，因为可能还有主仓信号

        # ========== 优先级3：极端仓止盈 ==========
        if self.extreme_position > 0:
            if best_bid >= EXTREME_SELL_PRICE:
                sell_qty = min(self.extreme_position, bid_vol)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    self.extreme_position = 0
                    self.extreme_entry = 0
                    return orders

        # ========== 主交易逻辑 ==========
        # 条件：未触发止损，且当前无持仓
        if not self.stop_loss_triggered and total_position == 0:
            # 买入信号：best_ask <= FV - BUY_DELTA
            if best_ask <= fv - BUY_DELTA:
                buy_qty = min(BUY_SIZE, ask_vol)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    self.entry_price = best_ask
                    return orders

        # ========== 卖出逻辑 ==========
        # 条件：有多头持仓（主仓或极端仓）
        if total_position > 0:
            # 卖出信号：best_bid >= FV + SELL_DELTA
            if best_bid >= fv + SELL_DELTA:
                sell_qty = min(total_position, bid_vol)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    self.entry_price = 0
                    self.extreme_position = 0
                    self.extreme_entry = 0
                    return orders

        return orders


# ============== 使用示例 ==============

if __name__ == "__main__":
    print("ASHIntegerStrategy 已定义")
    print()
    print("参数配置:")
    print(f"  FV窗口: {FV_WINDOW}")
    print(f"  买入触发: best_ask <= FV - {BUY_DELTA}")
    print(f"  卖出触发: best_bid >= FV + {SELL_DELTA}")
    print(f"  持仓上限: {MAX_POSITION}")
    print(f"  止损阈值: {STOP_LOSS_DELTA}点")
    print()
    print("极端价格捕捉:")
    print(f"  买入条件: best_ask <= {EXTREME_PRICE}")
    print(f"  止盈条件: best_bid >= {EXTREME_SELL_PRICE}")
    print(f"  止损条件: best_bid <= {EXTREME_STOP_PRICE}")