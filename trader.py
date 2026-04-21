"""
量化交易策略教程 - 完整注释版
=====================================

本文件实现了两套完整的量化交易策略：

1. ASH_COATED_OSMIUM (ASH) - 趋势中性均值回归策略
   适用于价格围绕稳定价值中枢波动的品种。
   核心思想：当价格偏离"公平价值"时，低买高卖期待价格回归。

2. INTARIAN_PEPPER_ROOT (INTARIAN) - 趋势跟踪 + 短线 scalp 策略
   适用于有明显趋势的品种。
   核心思想：追随趋势方向建仓，用移动止损保护利润，震荡时做短线交易。

适用学员：具备 Python 基础，了解金融市场基本概念（买/卖订单、持仓、限价单等）。

关键概念速查：
- Position（持仓）：当前持有的合约数量，正=多头，负=空头
- Order Depth（订单簿）：市场上所有待成交的买卖订单
- Fair Value（公平价值）：策略认为的"真实价格"
- EMA：指数移动平均线，用于平滑价格序列
- MA Crossover：短期均线与长期均线的交叉，作为趋势信号
"""

import json
from typing import Dict, List, Tuple

from datamodel import OrderDepth, Order, TradingState


# ============================================================
# 策略常量定义
# ============================================================

# 两个交易品种的交易代码（由交易所指定）
PRODUCT_ASH = "ASH_COATED_OSMIUM"
PRODUCT_INTARIAN = "INTARIAN_PEPPER_ROOT"

# 持仓上限：每只产品最多持有 80 手
# 这是交易所规定的仓位限制，防止过度集中风险
POSITION_LIMITS = {
    PRODUCT_ASH: 80,
    PRODUCT_INTARIAN: 80,
}

# ---------- ASH 策略参数 ----------

# ASH 的基本面价值锚定价格
# 策略认为 ASH 有一个内在价值中枢，长期来看价格会围绕它波动
ASH_ANCHOR = 10000.0

# EMA 平滑系数 (0 < alpha < 1)
# alpha 越大，EMA 越敏感，跟随近期价格越快
# 此处 alpha=0.08 偏低，说明我们更看重历史价格而非短期波动
ASH_EMA_ALPHA = 0.08

# 公平价值的构成权重
# 0.75 * 锚定价格 + 0.25 * EMA = 公平价值
# 即：75% 相信基本面价值，25% 相信市场定价
ASH_ANCHOR_WEIGHT = 0.75

# 持仓偏移因子：每持有 1 单位多头，公平价值下调一点
# 目的：多头持仓越多，越不愿意买（防止过度追高）
ASH_POSITION_SKEW = 0.1

# 主动吃单（take）的价差收益阈值
# 0 表示不吃单（只做市商挂单）
ASH_TAKE_EDGE = 0.0

# 用于判断市场订单簿稀密的阈值
# 当买卖价差 > 8 时认为市场较稀薄，此时使用更宽的挂单间距
ASH_WIDE_SPREAD_CUTOFF = 8.0

# 挂单间距参数（单位：价格单位）
# inner_width：最优档位与公平价值的距离
# outer_width = inner_width + ASH_OUTER_STEP：第二档位与公平价值的距离
ASH_INNER_WIDTH_WIDE = 1.0   # 市场稀薄时（spread 大）用这个
ASH_INNER_WIDTH_TIGHT = 2.0  # 市场活跃时（spread 小）用这个
ASH_OUTER_STEP = 2.0         # 内档到外档的距离

# 订单簿不平衡权重（当前设为 0，不使用）
# 如果设为非零值：买入方向压力越大，公平价值越高
ASH_IMBALANCE_WEIGHT = 0.0

# 中性市场下每档挂单数量
ASH_NEUTRAL_SIZE_1 = 8  # 内档数量
ASH_NEUTRAL_SIZE_2 = 4  # 外档数量

# 持仓阈值：超过这个绝对值后开始调整挂单方向
# 例如 temp_pos > 40（多头 40 手以上）时，减少买入、加大卖出
ASH_INVENTORY_THRESHOLD = 40

# ---------- INTARIAN 策略参数 ----------

# 移动平均线周期
INT_MA_SHORT = 5    # 短期均线：最近 5 个价格均值
INT_MA_LONG = 8     # 长期均线：最近 8 个价格均值

# 用于判断趋势的历史数据长度
INT_LOOKBACK = 20   # 取最近 20 个价格计算历史高点

# 建仓/加仓需要连续满足趋势条件的次数
INT_ENTRY_CONSEC = 1     # 首次建仓：连续 N 次趋势向上才入场
INT_ADD_CONSEC = 1       # 加仓：连续 N 次趋势向上才加仓

# 每次下单的数量
INT_FIRST_SIZE = 20   # 首次建仓量
INT_ADD_SIZE = 20    # 加仓量

# 止损参数
INT_STOP_LOSS_PCT = 0.015      # 固定止损：亏损超过 1.5% 则清仓
INT_TRAILING_STOP_PCT = 0.025  # 移动止损：从最高点下跌 2.5% 则清仓

# Scalp（短线交易）参数
INT_SCALP_SIZE = 20       # 每次 scalp 交易的手数
INT_VOL_WINDOW = 20       # 波动率计算窗口

# 波动率信号的倍数阈值
INT_PARTIAL_TAKE_VOL_MULT = 1.2   # 止盈阈值 = 短均线 + 1.2 * 波动率
INT_REBUY_VOL_MULT = 0.35          # 回补阈值 = 短均线 + 0.35 * 波动率


# ============================================================
# 辅助函数
# ============================================================

def best_bid_ask(order_depth: OrderDepth) -> Tuple[int, int]:
    """
    从订单簿提取最优买卖报价。

    Args:
        order_depth: 订单簿对象，包含 buy_orders（买入订单字典）
                     和 sell_orders（卖出订单字典）

    Returns:
        (最优买价, 最优卖价)
        最优买价 = 所有买入订单中的最高价（愿意出最高价的人）
        最优卖价 = 所有卖出订单中的最低价（愿意以最低价卖的人）

    Example:
        order_depth.buy_orders = {100: 5, 101: 3}  # 有人愿以 100 买 5 手，101 买 3 手
        order_depth.sell_orders = {102: 4, 103: 2} # 有人愿以 102 卖 4 手，103 卖 2 手
        -> best_bid_ask = (101, 102)
    """
    return max(order_depth.buy_orders), min(order_depth.sell_orders)


def mid_price(order_depth: OrderDepth) -> float:
    """
    计算中间价（最优买价与最优卖价的平均值）。

    中间价是评估当前市场定价的简单方法，
    优点是对单笔大单不敏感，缺点是没考虑订单量差异。
    """
    best_bid, best_ask = best_bid_ask(order_depth)
    return (best_bid + best_ask) / 2


def clamp(value: float, lower: float, upper: float) -> float:
    """
    将数值限制在 [lower, upper] 区间内。

    等价于: lower if value < lower else upper if value > upper else value
    用于防止下单价格超出合理范围。
    """
    return max(lower, min(upper, value))


def take_sell_capacity(position: int, limit: int) -> int:
    """
    计算当前还能卖出多少手（不违反持仓限制）。

    公式：position + limit
    例如：position=30, limit=80 -> 还能卖 110 手（但实际受限于空方订单量）
    注意：position 为负表示当前持有空头，position + limit 即为可持有的最大多头

    Args:
        position: 当前持仓（正=多头，负=空头）
        limit: 持仓上限（固定为 80）
    """
    return position + limit


# ============================================================
# Trader 主类
# ============================================================

class Trader:
    """
    量化交易策略引擎。

    管理两个产品的交易逻辑、状态持久化和订单生成。
    每个交易回合（run 方法被调用一次），接收市场快照，
    输出 orders（订单列表）和更新后的状态数据。
    """

    def __init__(self) -> None:
        """初始化 Trader，加载持仓上限配置。"""
        self.limits = POSITION_LIMITS

    def bounded_append(
        self, history: List[float], price: float, max_len: int = 100
    ) -> List[float]:
        """
        将新价格追加到历史价格序列，并截断到最大长度。

        防止历史数据无限膨胀，只保留最近 max_len 个价格点。
        这是计算移动平均线的前提。

        Args:
            history: 历史价格列表
            price: 当前中间价
            max_len: 历史数据最大长度（默认 100）

        Returns:
            更新后的历史价格列表
        """
        history.append(price)
        if len(history) > max_len:
            # 保留最新的 max_len 个数据点
            history = history[-max_len:]
        return history

    def load_data(self, state: TradingState) -> Dict:
        """
        从交易状态中加载持久化数据。

        TradingState 包含一个 traderData 字段，用于在回合之间传递状态。
        我们用 JSON 格式序列化/反序列化状态字典。

        Args:
            state: 当前交易状态

        Returns:
            解析后的状态字典，如果无数据则返回空字典
        """
        raw = getattr(state, "traderData", "")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            # 如果解析失败（如格式损坏），返回空字典，不中断交易
            return {}

    def save_data(self, data: Dict) -> str:
        """
        将状态字典序列化为 JSON 字符串。

        使用紧凑格式（无空格）减少数据传输开销。
        """
        return json.dumps(data, separators=(",", ":"))

    def update_ema(
        self, prev: float | None, price: float, alpha: float
    ) -> float:
        """
        更新指数移动平均线（EMA）。

        EMA 公式：EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}

        特点：
        - alpha 越大，近期价格权重越高，EMA 越敏感
        - alpha 越小，EMA 越平滑，适合抓大趋势

        与简单移动平均（SMA）的区别：
        SMA = (P1 + P2 + ... + Pn) / n，对所有历史数据一视同仁
        EMA 更重视最近的数据，因此响应更快

        Args:
            prev: 上一个周期的 EMA 值，None 表示无历史数据（直接用当前价初始化）
            price: 当前价格
            alpha: 平滑系数 (0 < alpha < 1)

        Returns:
            更新后的 EMA 值
        """
        if prev is None:
            return price
        return alpha * price + (1 - alpha) * prev

    # --------------------------------------------------------
    # ASH 策略：均值回归做市商
    # --------------------------------------------------------
    def trade_ash(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        state_data: Dict,
    ) -> Tuple[List[Order], Dict]:
        """
        ASH 产品的均值回归做市策略。

        策略原理：
        -----------
        1. 估计"公平价值"（fair value）：
           75% 相信基本面锚定价格（10000）+ 25% 跟随市场 EMA
           这样既有一个价值基准，又不过度固执

        2. 如果市场上有人愿意以低于 fair value 的价格卖，
           则主动买入（take sell side）；反之主动卖出（take buy side）

        3. 同时在 fair value 附近挂出买卖单（make market），
           等待对手方来成交，从而赚取买卖价差

        4. 根据持仓方向调整挂单价格：
           - 多头持仓多 -> 降低买入报价，减少买入（避免追高）
           - 空头持仓多 -> 提高卖出报价，减少卖出（避免杀跌）

        Args:
            product: 交易品种代码（"ASH_COATED_OSMIUM"）
            order_depth: 当前订单簿（所有未成交订单）
            position: 当前持仓（正=多头，负=空头）
            state_data: 从上一轮继承的状态（含 ash_ema）

        Returns:
            (orders, state_out): 本轮生成的订单列表，以及需要持久化的状态
        """
        orders: List[Order] = []
        limit = self.limits[product]

        # ----- 从订单簿提取市场信息 -----
        best_bid, best_ask = best_bid_ask(order_depth)
        spread = best_ask - best_bid          # 买卖价差，越小市场越活跃
        mid = (best_bid + best_ask) / 2       # 中间价

        # 计算订单簿不平衡度
        # imbalance > 0 表示买方力量更强（可能推动价格上涨）
        # imbalance < 0 表示卖方力量更强（可能推动价格下跌）
        bid_volume = order_depth.buy_orders.get(best_bid, 0)
        ask_volume = -order_depth.sell_orders.get(best_ask, 0)  # 卖出量为负，取正
        imbalance = 0.0
        if bid_volume + ask_volume > 0:
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        # ----- 计算公平价值 -----
        # 更新 EMA（追踪市场价格趋势）
        ash_ema = self.update_ema(state_data.get("ash_ema"), mid, ASH_EMA_ALPHA)

        # 基础公平价值 = 权重 * 锚定价格 + 权重 * EMA
        base_fair = ASH_ANCHOR_WEIGHT * ASH_ANCHOR + (1 - ASH_ANCHOR_WEIGHT) * ash_ema

        # 调整公平价值：
        # 持仓越多，越不倾向于继续同方向开仓（抑制追涨杀跌）
        # 订单簿不平衡也会轻微影响公平价值（当前未启用）
        fair = base_fair - ASH_POSITION_SKEW * position + ASH_IMBALANCE_WEIGHT * imbalance

        # ----- 主动吃单（Take）逻辑 -----
        # 当价格足够优惠时，主动成交（以更低的价格买，或更高的价格卖）
        # ASH_TAKE_EDGE = 0，表示我们不在这一层主动出击（只做市商）
        buy_take_edge = ASH_TAKE_EDGE
        sell_take_edge = ASH_TAKE_EDGE
        temp_pos = position  # 模拟成交后的持仓变化

        # 遍历所有卖出订单（价格从低到高）
        # 如果某卖价 <= fair + buy_take_edge，说明这个卖价低于我们认为的公平价值，
        # 值得买入（预期价格会涨回来）
        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price <= fair + buy_take_edge and temp_pos < limit:
                # 尽可能多买，但不超过持仓上限
                quantity = min(-ask_volume, limit - temp_pos)
                if quantity > 0:
                    orders.append(Order(product, ask_price, quantity))
                    temp_pos += quantity
            else:
                # 一旦遇到不再优惠的价格，后面的更贵的也不会考虑了
                break

        # 遍历所有买入订单（价格从高到低）
        # 如果某买价 >= fair - sell_take_edge，值得卖出（预期价格会跌回来）
        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price >= fair - sell_take_edge and temp_pos > -limit:
                quantity = min(bid_volume, take_sell_capacity(temp_pos, limit))
                if quantity > 0:
                    orders.append(Order(product, bid_price, -quantity))
                    temp_pos -= quantity
            else:
                break

        # ----- 做市商（Make）挂单逻辑 -----
        # 剩余的可交易空间
        buy_room = limit - temp_pos     # 还能买多少手
        sell_room = temp_pos + limit    # 还能卖多少手（temp_pos 为负时更大）

        # 根据当前市场价差宽度决定挂单间距
        # 价差大（spread >= 8）-> 市场稀薄 -> 用宽间距（1.0）
        # 价差小（spread < 8） -> 市场活跃 -> 用紧间距（2.0）
        inner_width = ASH_INNER_WIDTH_WIDE if spread >= ASH_WIDE_SPREAD_CUTOFF else ASH_INNER_WIDTH_TIGHT
        outer_width = inner_width + ASH_OUTER_STEP  # 外档 = 内档 + 2

        # 计算四个挂单价格
        # buy_quote_1: 内档买入价 = min(最优买价+1, fair - inner_width)
        # sell_quote_1: 内档卖出价 = max(最优卖价-1, fair + inner_width)
        # buy_quote_2: 外档买入价 = min(最优买价, fair - outer_width)
        # sell_quote_2: 外档卖出价 = max(最优卖价, fair + outer_width)
        buy_quote_1 = min(best_bid + 1, int(fair - inner_width))
        sell_quote_1 = max(best_ask - 1, int(fair + inner_width))
        buy_quote_2 = min(best_bid, int(fair - outer_width))
        sell_quote_2 = max(best_ask, int(fair + outer_width))

        # ----- 持仓控制：动态调整挂单方向和大小 -----
        # 目的：当持仓偏向一侧时，减少同向暴露，增加反向暴露
        if temp_pos > ASH_INVENTORY_THRESHOLD:
            # 多头太重（> 40手）：大幅减少买入，小幅增加卖出
            # -> 降低买入量到 2/1 手，增加卖出量到 8/4 手
            buy_size_1, buy_size_2 = 2, 1
            sell_size_1, sell_size_2 = 8, 4
        elif temp_pos < -ASH_INVENTORY_THRESHOLD:
            # 空头太重（< -40手）：大幅减少卖出，小幅增加买入
            buy_size_1, buy_size_2 = 8, 4
            sell_size_1, sell_size_2 = 2, 1
        else:
            # 中性市场：内档 8 手，外档 4 手
            buy_size_1, buy_size_2 = ASH_NEUTRAL_SIZE_1, ASH_NEUTRAL_SIZE_2
            sell_size_1, sell_size_2 = ASH_NEUTRAL_SIZE_1, ASH_NEUTRAL_SIZE_2

        # 发出内档挂单（更接近中间价，成交概率更高）
        if buy_room > 0:
            orders.append(Order(product, buy_quote_1, min(buy_size_1, buy_room)))
        if sell_room > 0:
            orders.append(Order(product, sell_quote_1, -min(sell_size_1, sell_room)))

        # 发出外档挂单（距离中间价更远，成交概率低但收益更高）
        remaining_buy = max(0, buy_room - min(buy_size_1, buy_room))
        remaining_sell = max(0, sell_room - min(sell_size_1, sell_room))
        if remaining_buy > 0:
            orders.append(Order(product, buy_quote_2, min(buy_size_2, remaining_buy)))
        if remaining_sell > 0:
            orders.append(Order(product, sell_quote_2, -min(sell_size_2, remaining_sell)))

        # 保存 EMA 状态到 state_out，以便下一轮使用
        state_out = {"ash_ema": ash_ema}
        return orders, state_out

    # --------------------------------------------------------
    # INTARIAN 策略：趋势跟踪 + 移动止损 + Scalping
    # --------------------------------------------------------
    def trade_intarian(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        state_data: Dict,
    ) -> Tuple[List[Order], Dict]:
        """
        INTARIAN 产品的趋势跟踪 + scalp 策略。

        策略原理（三种模式）：
        -----------
        【模式 A：建仓】
        当短期均线 > 长期均线（上升趋势）时，在回调位置买入。
        等待连续 N 次确认趋势后才建仓，避免假突破。

        【模式 B：加仓】
        建仓后如果趋势持续且价格突破历史高点，继续加仓。

        【模式 C：止损保护】
        - 固定止损：从买入价下跌 1.5% 则清仓
        - 移动止损：从持仓期间的最高点下跌 2.5% 则清仓
          这确保了：浮亏时及时止损，浮盈时"让利润奔跑"

        【模式 D：Scalp 短线】
        当价格短时快速上涨（超过波动率阈值）时，卖出部分持仓锁定利润。
        之后等待价格回落（回补阈值）时再买回，保持持仓规模。
        这一层是额外的收益增强，不改变整体趋势跟踪方向。

        状态机（scalp_state）：
        - "neutral": 正常持仓
        - "waiting_rebuy": scalp 后等待回补

        Args:
            product: 交易品种代码（"INTARIAN_PEPPER_ROOT"）
            order_depth: 当前订单簿
            position: 当前持仓
            state_data: 从上一轮继承的状态（含 history, entry_price, highest_price,
                       consecutive_uptrend, scalp_state）

        Returns:
            (orders, state_out): 订单列表和新状态
        """
        orders: List[Order] = []
        limit = self.limits[product]

        # ----- 从订单簿提取市场信息 -----
        best_bid, best_ask = best_bid_ask(order_depth)
        mid = (best_bid + best_ask) / 2

        # ----- 加载/更新历史数据 -----
        # history：保存最近中间价序列，用于计算移动平均
        history = list(state_data.get("history", []))
        history = self.bounded_append(history, mid)

        # 从状态中恢复交易关键变量
        entry_price = state_data.get("entry_price", 0.0)      # 首次建仓价格（用于计算止损）
        highest_price = state_data.get("highest_price", 0.0)  # 持仓期间最高价（用于移动止损）
        consecutive_uptrend = state_data.get("consecutive_uptrend", 0)  # 连续趋势向上次数
        scalp_state = state_data.get("scalp_state", "neutral")  # scalp 状态机

        # ----- 数据不足时直接返回（避免计算无效的 MA） -----
        if len(history) < max(INT_MA_SHORT, INT_MA_LONG):
            state_out = {
                "history": history,
                "entry_price": entry_price,
                "highest_price": highest_price,
                "consecutive_uptrend": consecutive_uptrend,
                "scalp_state": scalp_state,
            }
            return orders, state_out

        # ----- 计算技术指标 -----
        # 移动平均线
        short_ma = sum(history[-INT_MA_SHORT:]) / INT_MA_SHORT
        long_ma = sum(history[-INT_MA_LONG:]) / INT_MA_LONG

        # 趋势判断：短期 MA 在长期 MA 之上 = 上升趋势
        is_uptrend = short_ma > long_ma

        # 波动率估算：使用最近 N 个价格的高低价差
        # 为什么不使用标准差？因为简单高效，对快速变化更敏感
        vol_slice = history[-INT_VOL_WINDOW:] if len(history) >= INT_VOL_WINDOW else history
        recent_vol = max(vol_slice) - min(vol_slice) if len(vol_slice) >= 2 else 0.0

        # 基于波动率的止盈和回补阈值
        # 止盈阈值：超过此价格则认为短期过热，可能回调
        take_threshold = short_ma + INT_PARTIAL_TAKE_VOL_MULT * recent_vol
        # 回补阈值：价格回到此水平时，认为可以重新买入
        rebuy_threshold = short_ma + INT_REBUY_VOL_MULT * recent_vol

        # 价格突破判断：当前价格创出 N 周期内新高（容许 0.1% 的浮点误差）
        lookback_slice = history[-INT_LOOKBACK:] if len(history) >= INT_LOOKBACK else history
        lookback_high = max(lookback_slice)
        is_breakout = mid > lookback_high * 0.999

        # 趋势连续计数（用于过滤假信号）
        # 例如：要求连续 2 次趋势向上才建仓，可以避免被单次逆势波动骗
        if is_uptrend:
            consecutive_uptrend += 1
        else:
            consecutive_uptrend = 0

        available = limit - position  # 还能开多少手

        # ==================== 决策逻辑 ====================

        # 情况 1：当前无持仓 -> 尝试建仓
        if position == 0:
            if is_uptrend and consecutive_uptrend >= INT_ENTRY_CONSEC:
                volume = min(available, INT_FIRST_SIZE)
                if volume > 0:
                    # 以市场价（最优卖价）买入
                    orders.append(Order(product, int(best_ask), volume))
                    entry_price = mid
                    highest_price = mid
                    scalp_state = "neutral"

        # 情况 2：当前有多头持仓但未满 -> 考虑加仓 或 scalp 止盈
        elif 0 < position < limit:
            # 加仓条件：上升趋势 + 突破新高 + 连续趋势确认
            if is_uptrend and is_breakout and consecutive_uptrend >= INT_ADD_CONSEC:
                volume = min(available, INT_ADD_SIZE)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    highest_price = max(highest_price, mid)

            # Scalp 止盈条件：
            # - 上升趋势（但趋势仍在，不做逆势操作）
            # - 有足够持仓（> scalp_size）
            # - 当前不在等待回补状态
            # - 价格超过止盈阈值（短期过热）
            if (
                is_uptrend
                and recent_vol > 0
                and position > INT_SCALP_SIZE
                and scalp_state != "waiting_rebuy"
                and mid >= take_threshold
            ):
                trim_qty = min(INT_SCALP_SIZE, position)
                if trim_qty > 0:
                    # 以市场价（最优买价）卖出部分持仓
                    orders.append(Order(product, int(best_bid), -trim_qty))
                    scalp_state = "waiting_rebuy"

        # 情况 3：当前有多头持仓（任意规模）-> 止损检查 + 移动止损 + scalp/回补
        elif position > 0:
            # 更新持仓期间的最高价
            highest_price = max(highest_price, mid)

            # 固定止损：亏损超过 INT_STOP_LOSS_PCT（1.5%）则清仓
            if entry_price > 0 and mid < entry_price * (1 - INT_STOP_LOSS_PCT):
                orders.append(Order(product, int(best_bid), -position))
                entry_price = 0.0
                highest_price = 0.0
                consecutive_uptrend = 0
                scalp_state = "neutral"
            else:
                # 移动止损：最高点下跌 2.5% 则清仓
                # 这允许利润在上涨时持续增长，但一旦反转超过阈值立刻退出
                trailing_stop = highest_price * (1 - INT_TRAILING_STOP_PCT) if highest_price > 0 else 0.0
                if trailing_stop > 0 and mid < trailing_stop:
                    orders.append(Order(product, int(best_bid), -position))
                    entry_price = 0.0
                    highest_price = 0.0
                    consecutive_uptrend = 0
                    scalp_state = "neutral"
                else:
                    # 没有触发止损 -> 检查是否需要 scalp 止盈
                    if (
                        is_uptrend
                        and recent_vol > 0
                        and position > INT_SCALP_SIZE
                        and scalp_state != "waiting_rebuy"
                        and mid >= take_threshold
                    ):
                        # 卖出部分持仓（止盈）
                        trim_qty = min(INT_SCALP_SIZE, position)
                        if trim_qty > 0:
                            orders.append(Order(product, int(best_bid), -trim_qty))
                            scalp_state = "waiting_rebuy"

                    # Scalp 后等待回补（价格回落到合理区间再买回）
                    elif (
                        scalp_state == "waiting_rebuy"
                        and is_uptrend  # 趋势仍在（不要逆势补回）
                        and recent_vol > 0
                        and mid <= rebuy_threshold  # 价格回到合理区间
                        and available > 0  # 还有仓位可用
                    ):
                        rebuy_qty = min(INT_SCALP_SIZE, available)
                        if rebuy_qty > 0:
                            orders.append(Order(product, int(best_ask), rebuy_qty))
                            scalp_state = "neutral"

        # ----- 保存所有状态到 state_out -----
        state_out = {
            "history": history,
            "entry_price": entry_price,
            "highest_price": highest_price,
            "consecutive_uptrend": consecutive_uptrend,
            "scalp_state": scalp_state,
        }
        return orders, state_out

    # ============================================================
    # run：主回合函数
    # ============================================================

    def run(self, state: TradingState):
        """
        交易引擎的主入口，每个时间步调用一次。

        负责：
        1. 加载上一轮持久化的状态
        2. 遍历所有交易品种，执行对应的交易策略
        3. 返回订单列表、转换次数（此版本不用）和新的持久化状态

        Args:
            state: 包含当前市场快照（order_depths）、持仓（position）等

        Returns:
            (result, conversions, trader_data)
            - result: Dict[产品代码, 订单列表]，包含每个产品的交易指令
            - conversions: 转换次数（此处设为 0，本策略不涉及）
            - trader_data: 序列化后的状态数据，供下一轮使用
        """
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # 从 state.traderData 恢复上一轮保存的状态
        data = self.load_data(state)
        ash_state = data.get("ash", {})
        intarian_state = data.get("intarian", {})

        # 遍历所有有报价的产品
        for product, order_depth in state.order_depths.items():
            # 跳过没有买卖订单的产品（无法交易）
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            # 获取当前持仓（无持仓则为 0）
            position = state.position.get(product, 0)

            # 根据产品选择对应的交易策略
            if product == PRODUCT_ASH:
                orders, ash_state = self.trade_ash(
                    product, order_depth, position, ash_state
                )
                result[product] = orders
            elif product == PRODUCT_INTARIAN:
                orders, intarian_state = self.trade_intarian(
                    product, order_depth, position, intarian_state
                )
                result[product] = orders

        # 将两个产品的状态打包并序列化
        trader_data = self.save_data({
            "ash": ash_state,
            "intarian": intarian_state,
        })
        return result, conversions, trader_data


# ============================================================
# 回测入口（仅在直接运行本文件时执行）
# ============================================================

if __name__ == "__main__":
    """
    直接运行 trader.py 即可执行回测。

    回测流程：
    1. 启动 Rust 回测引擎（需要提前编译）
    2. 加载 trader.py 作为交易逻辑
    3. 在历史数据集（round2）上模拟运行
    4. 输出结果并保存到 backtest_result.json
    """
    import os
    import sys
    import subprocess
    import json

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 回测器二进制文件路径
    cache_dir = os.path.expanduser("~/Library/Caches/rust_backtester/target/debug")
    backtester_bin = os.path.join(cache_dir, "rust_backtester")

    # 回测器源代码目录
    backtester_dir = os.path.expanduser("~/Downloads/prosperity_rust_backtester")

    # 检查回测器是否已编译
    if not os.path.exists(backtester_bin):
        print(
            f"错误：找不到回测器。请先运行：cd {backtester_dir} && make backtest"
        )
        sys.exit(1)

    # 运行回测
    result = subprocess.run(
        [backtester_bin,
         '--trader', f'{script_dir}/trader.py',
         '--dataset', 'round2',
         '--products', 'full'],
        capture_output=True,
        text=True,
        cwd=backtester_dir
    )
    result_file = result.stdout + result.stderr
    print(result_file)

    # ----- 解析回测结果 -----
    lines = result_file.strip().split("\n")
    summary = {
        "trader": "trader.py",
        "dataset": "round2",
        "raw_output": result_file,
    }

    # TOTAL 行格式：TOTAL  -    TICKS    TRADES        PNL  -
    # 列索引：           0      1       2           3         4  5
    for line in lines:
        if line.startswith("TOTAL"):
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "TOTAL" or p == "-":
                    continue
                try:
                    val = float(p.replace(",", ""))
                except ValueError:
                    continue
                # TRADES 在第 3 列（0-indexed）
                if i == 3 and 1000 < val < 10000:
                    summary["total_trades"] = int(val)
                # PNL 在第 4 列（0-indexed）
                if i == 4 and val > 100000:
                    summary["total_pnl"] = val

        # 解析各产品 PNL
        if "ASH_COATED_OSMIUM" in line:
            parts = line.split()
            summary["ash_pnl"] = float(parts[-1].replace(",", ""))
        if "INTARIAN_PEPPER_ROOT" in line:
            parts = line.split()
            summary["intarian_pnl"] = float(parts[-1].replace(",", ""))

    # ----- 保存结果到 JSON -----
    output_path = os.path.join(script_dir, "backtest_result.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n回测结果已保存到: {output_path}")
