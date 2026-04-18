import json
from typing import Dict, List, Tuple

from datamodel import OrderDepth, Order, TradingState


PRODUCT_ASH = "ASH_COATED_OSMIUM"
PRODUCT_INTARIAN = "INTARIAN_PEPPER_ROOT"

# Official Round 1 limits are 80 lots for both products.
POSITION_LIMITS = {
    PRODUCT_ASH: 80,
    PRODUCT_INTARIAN: 80,
}

ASH_ANCHOR = 10000.0

ASH_EMA_ALPHA = 0.08
ASH_ANCHOR_WEIGHT = 0.75
ASH_POSITION_SKEW = 0.1
ASH_TAKE_EDGE = 0.0
ASH_WIDE_SPREAD_CUTOFF = 8.0
ASH_INNER_WIDTH_WIDE = 1.0
ASH_INNER_WIDTH_TIGHT = 2.0
ASH_OUTER_STEP = 2.0
ASH_IMBALANCE_WEIGHT = 0.0
ASH_NEUTRAL_SIZE_1 = 8
ASH_NEUTRAL_SIZE_2 = 4
ASH_INVENTORY_THRESHOLD = 40

INT_MA_SHORT = 5
INT_MA_LONG = 8
INT_LOOKBACK = 20
INT_ENTRY_CONSEC = 1
INT_ADD_CONSEC = 1
INT_FIRST_SIZE = 20
INT_ADD_SIZE = 20
INT_STOP_LOSS_PCT = 0.015
INT_TRAILING_STOP_PCT = 0.025
INT_SCALP_SIZE = 20
INT_VOL_WINDOW = 20
INT_PARTIAL_TAKE_VOL_MULT = 1.2
INT_REBUY_VOL_MULT = 0.35


def best_bid_ask(order_depth: OrderDepth) -> Tuple[int, int]:
    return max(order_depth.buy_orders), min(order_depth.sell_orders)


def mid_price(order_depth: OrderDepth) -> float:
    best_bid, best_ask = best_bid_ask(order_depth)
    return (best_bid + best_ask) / 2


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def take_sell_capacity(position: int, limit: int) -> int:
    return position + limit


class Trader:
    def __init__(self) -> None:
        self.limits = POSITION_LIMITS

    def bounded_append(self, history: List[float], price: float, max_len: int = 100) -> List[float]:
        history.append(price)
        if len(history) > max_len:
            history = history[-max_len:]
        return history

    def load_data(self, state: TradingState) -> Dict:
        raw = getattr(state, "traderData", "")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def save_data(self, data: Dict) -> str:
        return json.dumps(data, separators=(",", ":"))

    def update_ema(self, prev: float, price: float, alpha: float) -> float:
        if prev is None:
            return price
        return alpha * price + (1 - alpha) * prev

    def trade_ash(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        state_data: Dict,
    ) -> Tuple[List[Order], Dict]:
        orders: List[Order] = []
        limit = self.limits[product]

        best_bid, best_ask = best_bid_ask(order_depth)
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        bid_volume = order_depth.buy_orders.get(best_bid, 0)
        ask_volume = -order_depth.sell_orders.get(best_ask, 0)
        imbalance = 0.0
        if bid_volume + ask_volume > 0:
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        ash_ema = self.update_ema(state_data.get("ash_ema"), mid, ASH_EMA_ALPHA)
        base_fair = ASH_ANCHOR_WEIGHT * ASH_ANCHOR + (1 - ASH_ANCHOR_WEIGHT) * ash_ema
        fair = base_fair - ASH_POSITION_SKEW * position + ASH_IMBALANCE_WEIGHT * imbalance

        # ASH is a stable mean-reversion product, so we can be fairly aggressive
        # when quotes are close to fair and then replenish inventory passively.
        buy_take_edge = ASH_TAKE_EDGE
        sell_take_edge = ASH_TAKE_EDGE
        temp_pos = position

        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price <= fair + buy_take_edge and temp_pos < limit:
                quantity = min(-ask_volume, limit - temp_pos)
                if quantity > 0:
                    orders.append(Order(product, ask_price, quantity))
                    temp_pos += quantity
            else:
                break

        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price >= fair - sell_take_edge and temp_pos > -limit:
                quantity = min(bid_volume, take_sell_capacity(temp_pos, limit))
                if quantity > 0:
                    orders.append(Order(product, bid_price, -quantity))
                    temp_pos -= quantity
            else:
                break

        buy_room = limit - temp_pos
        sell_room = temp_pos + limit

        inner_width = ASH_INNER_WIDTH_WIDE if spread >= ASH_WIDE_SPREAD_CUTOFF else ASH_INNER_WIDTH_TIGHT
        outer_width = inner_width + ASH_OUTER_STEP

        buy_quote_1 = min(best_bid + 1, int(fair - inner_width))
        sell_quote_1 = max(best_ask - 1, int(fair + inner_width))
        buy_quote_2 = min(best_bid, int(fair - outer_width))
        sell_quote_2 = max(best_ask, int(fair + outer_width))

        if temp_pos > ASH_INVENTORY_THRESHOLD:
            buy_size_1, buy_size_2 = 2, 1
            sell_size_1, sell_size_2 = 8, 4
        elif temp_pos < -ASH_INVENTORY_THRESHOLD:
            buy_size_1, buy_size_2 = 8, 4
            sell_size_1, sell_size_2 = 2, 1
        else:
            buy_size_1, buy_size_2 = ASH_NEUTRAL_SIZE_1, ASH_NEUTRAL_SIZE_2
            sell_size_1, sell_size_2 = ASH_NEUTRAL_SIZE_1, ASH_NEUTRAL_SIZE_2

        if buy_room > 0:
            orders.append(Order(product, buy_quote_1, min(buy_size_1, buy_room)))
        if sell_room > 0:
            orders.append(Order(product, sell_quote_1, -min(sell_size_1, sell_room)))

        remaining_buy = max(0, buy_room - min(buy_size_1, buy_room))
        remaining_sell = max(0, sell_room - min(sell_size_1, sell_room))
        if remaining_buy > 0:
            orders.append(Order(product, buy_quote_2, min(buy_size_2, remaining_buy)))
        if remaining_sell > 0:
            orders.append(Order(product, sell_quote_2, -min(sell_size_2, remaining_sell)))

        state_out = {
            "ash_ema": ash_ema,
        }
        return orders, state_out

    def trade_intarian(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        state_data: Dict,
    ) -> Tuple[List[Order], Dict]:
        orders: List[Order] = []
        limit = self.limits[product]

        best_bid, best_ask = best_bid_ask(order_depth)
        mid = (best_bid + best_ask) / 2

        history = list(state_data.get("history", []))
        history = self.bounded_append(history, mid)

        entry_price = state_data.get("entry_price", 0.0)
        highest_price = state_data.get("highest_price", 0.0)
        consecutive_uptrend = state_data.get("consecutive_uptrend", 0)
        scalp_state = state_data.get("scalp_state", "neutral")

        if len(history) < max(INT_MA_SHORT, INT_MA_LONG):
            state_out = {
                "history": history,
                "entry_price": entry_price,
                "highest_price": highest_price,
                "consecutive_uptrend": consecutive_uptrend,
                "scalp_state": scalp_state,
            }
            return orders, state_out

        short_ma = sum(history[-INT_MA_SHORT:]) / INT_MA_SHORT
        long_ma = sum(history[-INT_MA_LONG:]) / INT_MA_LONG
        is_uptrend = short_ma > long_ma
        vol_slice = history[-INT_VOL_WINDOW:] if len(history) >= INT_VOL_WINDOW else history
        recent_vol = max(vol_slice) - min(vol_slice) if len(vol_slice) >= 2 else 0.0
        take_threshold = short_ma + INT_PARTIAL_TAKE_VOL_MULT * recent_vol
        rebuy_threshold = short_ma + INT_REBUY_VOL_MULT * recent_vol

        lookback_slice = history[-INT_LOOKBACK:] if len(history) >= INT_LOOKBACK else history
        lookback_high = max(lookback_slice)
        is_breakout = mid > lookback_high * 0.999

        if is_uptrend:
            consecutive_uptrend += 1
        else:
            consecutive_uptrend = 0

        available = limit - position

        if position == 0:
            if is_uptrend and consecutive_uptrend >= INT_ENTRY_CONSEC:
                volume = min(available, INT_FIRST_SIZE)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    entry_price = mid
                    highest_price = mid
                    scalp_state = "neutral"
        elif 0 < position < limit:
            if is_uptrend and is_breakout and consecutive_uptrend >= INT_ADD_CONSEC:
                volume = min(available, INT_ADD_SIZE)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    highest_price = max(highest_price, mid)

            if (
                is_uptrend
                and recent_vol > 0
                and position > INT_SCALP_SIZE
                and scalp_state != "waiting_rebuy"
                and mid >= take_threshold
            ):
                trim_qty = min(INT_SCALP_SIZE, position)
                if trim_qty > 0:
                    orders.append(Order(product, int(best_bid), -trim_qty))
                    scalp_state = "waiting_rebuy"

        elif position > 0:
            highest_price = max(highest_price, mid)

            if entry_price > 0 and mid < entry_price * (1 - INT_STOP_LOSS_PCT):
                orders.append(Order(product, int(best_bid), -position))
                entry_price = 0.0
                highest_price = 0.0
                consecutive_uptrend = 0
                scalp_state = "neutral"
            else:
                trailing_stop = highest_price * (1 - INT_TRAILING_STOP_PCT) if highest_price > 0 else 0.0
                if trailing_stop > 0 and mid < trailing_stop:
                    orders.append(Order(product, int(best_bid), -position))
                    entry_price = 0.0
                    highest_price = 0.0
                    consecutive_uptrend = 0
                    scalp_state = "neutral"
                else:
                    if (
                        is_uptrend
                        and recent_vol > 0
                        and position > INT_SCALP_SIZE
                        and scalp_state != "waiting_rebuy"
                        and mid >= take_threshold
                    ):
                        trim_qty = min(INT_SCALP_SIZE, position)
                        if trim_qty > 0:
                            orders.append(Order(product, int(best_bid), -trim_qty))
                            scalp_state = "waiting_rebuy"
                    elif (
                        scalp_state == "waiting_rebuy"
                        and is_uptrend
                        and recent_vol > 0
                        and mid <= rebuy_threshold
                        and available > 0
                    ):
                        rebuy_qty = min(INT_SCALP_SIZE, available)
                        if rebuy_qty > 0:
                            orders.append(Order(product, int(best_ask), rebuy_qty))
                            scalp_state = "neutral"

        state_out = {
            "history": history,
            "entry_price": entry_price,
            "highest_price": highest_price,
            "consecutive_uptrend": consecutive_uptrend,
            "scalp_state": scalp_state,
        }
        return orders, state_out

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        data = self.load_data(state)

        ash_state = data.get("ash", {})
        intarian_state = data.get("intarian", {})

        for product, order_depth in state.order_depths.items():
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            position = state.position.get(product, 0)

            if product == PRODUCT_ASH:
                orders, ash_state = self.trade_ash(product, order_depth, position, ash_state)
                result[product] = orders
            elif product == PRODUCT_INTARIAN:
                orders, intarian_state = self.trade_intarian(
                    product, order_depth, position, intarian_state
                )
                result[product] = orders

        trader_data = self.save_data({
            "ash": ash_state,
            "intarian": intarian_state,
        })
        return result, conversions, trader_data
