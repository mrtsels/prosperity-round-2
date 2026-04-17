"""
测试首批仓位大小 - 打印所有结果
"""

import json
import csv
import os
import sys
from io import StringIO
from typing import List, Dict, Tuple

sys.path.insert(0, '/Users/minimx/Downloads/ROUND_2')
from datamodel import OrderDepth, TradingState, Order


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


def load_simulator_data(json_path: str) -> Tuple[List[dict], List[dict]]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    activities = data.get('activitiesLog', '')
    reader = csv.DictReader(StringIO(activities), delimiter=';')

    intarian_data = []
    for row in reader:
        if row['product'] == 'INTARIAN_PEPPER_ROOT' and row.get('mid_price'):
            mp = float(row['mid_price'])
            if mp > 0:
                intarian_data.append({
                    'timestamp': int(row['timestamp']),
                    'mid_price': mp,
                    'bid_price_1': row.get('bid_price_1', ''),
                    'bid_volume_1': row.get('bid_volume_1', ''),
                    'ask_price_1': row.get('ask_price_1', ''),
                    'ask_volume_1': row.get('ask_volume_1', ''),
                })

    return intarian_data, []


def create_order_depth(data_row: dict) -> OrderDepth:
    od = OrderDepth()
    if data_row.get('bid_price_1') and data_row.get('bid_volume_1'):
        try:
            od.buy_orders[int(data_row['bid_price_1'])] = int(data_row['bid_volume_1'])
        except:
            pass
    if data_row.get('ask_price_1') and data_row.get('ask_volume_1'):
        try:
            od.sell_orders[int(data_row['ask_price_1'])] = -int(data_row['ask_volume_1'])
        except:
            pass
    return od


class SimulatedExchange:
    def __init__(self, position_limit: int = 80):
        self.position_limit = position_limit
        self.position: Dict[str, int] = {'INTARIAN_PEPPER_ROOT': 0}
        self.cash: float = 0.0

    def reset(self):
        self.position = {p: 0 for p in self.position}
        self.cash = 0.0

    def execute_orders(self, orders: List[Order], order_depths: Dict[str, OrderDepth]) -> float:
        for order in orders:
            product = order.symbol
            od = order_depths.get(product)
            if od is None:
                continue

            qty = order.quantity
            if qty > 0:
                if od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                    available_qty = abs(od.sell_orders[best_ask])
                    exec_qty = min(qty, available_qty)
                    if exec_qty > 0:
                        self.position[product] += exec_qty
                        self.cash -= best_ask * exec_qty
            elif qty < 0:
                if od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    available_qty = od.buy_orders[best_bid]
                    exec_qty = min(abs(qty), available_qty)
                    if exec_qty > 0:
                        new_position = self.position[product] - exec_qty
                        if abs(new_position) <= self.position_limit:
                            self.position[product] = new_position
                            self.cash += best_bid * exec_qty
        return 0.0

    def get_final_value(self, intarian_data):
        value = self.cash
        for product, pos in self.position.items():
            if pos != 0 and intarian_data:
                value += pos * intarian_data[-1]['mid_price']
        return value


class MomentumStrategyWithConfig:
    def __init__(self, first_position=20, add_position=20, profit_take_pct=0.02,
                 stop_loss_pct=0.015, trailing_stop_pct=0.025, lookback=20):
        self.position_limit = 80
        self.first_position = first_position
        self.add_position = add_position
        self.profit_take_pct = profit_take_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.lookback = lookback
        self.price_history: Dict[str, List[float]] = {'INTARIAN_PEPPER_ROOT': []}
        self.entry_price: Dict[str, float] = {}
        self.highest_price: Dict[str, float] = {}
        self.consecutive_uptrend = 0
        self.has_taken_profit = False

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
        if len(history) < 5:
            return orders

        short_ma = sum(history[-5:]) / 5
        long_ma = sum(history[-10:]) / 10 if len(history) >= 10 else short_ma
        is_uptrend = short_ma > long_ma
        lookback_high = max(history[-self.lookback:]) if len(history) >= self.lookback else max(history)
        is_breakout = mid_price > lookback_high * 0.999

        if is_uptrend:
            self.consecutive_uptrend += 1
        else:
            self.consecutive_uptrend = 0

        available = self.position_limit - position

        if position == 0:
            self.has_taken_profit = False
            if is_uptrend and self.consecutive_uptrend >= 3:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, self.first_position)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    self.entry_price[product] = mid_price
                    self.highest_price[product] = mid_price
        elif 0 < position < self.position_limit:
            if is_uptrend and is_breakout and self.consecutive_uptrend >= 5:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, self.add_position)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
        elif position > 0:
            if mid_price > self.highest_price[product]:
                self.highest_price[product] = mid_price

            entry = self.entry_price[product]
            peak = self.highest_price[product]
            profit_pct = (mid_price - entry) / entry

            # 固定止盈
            if not self.has_taken_profit and profit_pct >= self.profit_take_pct:
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
                self.has_taken_profit = True
                return orders

            # 止损
            if mid_price < entry * (1 - self.stop_loss_pct):
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
                return orders

            # 移动止损
            trailing_stop = peak * (1 - self.trailing_stop_pct)
            if mid_price < trailing_stop:
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
                return orders
        return orders


def run_strategy(intarian_data, first_position, add_position, profit_take_pct, stop_loss_pct, trailing_stop_pct, lookback):
    exchange = SimulatedExchange()
    exchange.reset()
    momentum = MomentumStrategyWithConfig(
        first_position, add_position, profit_take_pct,
        stop_loss_pct, trailing_stop_pct, lookback
    )

    all_timestamps = sorted(set(int(r['timestamp']) for r in intarian_data))

    for timestamp in all_timestamps:
        intarian_row = next((r for r in intarian_data if int(r['timestamp']) == timestamp), None)
        if not intarian_row:
            continue

        order_depths = {'INTARIAN_PEPPER_ROOT': create_order_depth(intarian_row)}

        state = TradingState(
            traderData='',
            timestamp=timestamp,
            listings={},
            order_depths=order_depths,
            own_trades={},
            market_trades={},
            position=dict(exchange.position),
            observations=None
        )

        orders = momentum.signal(state, 'INTARIAN_PEPPER_ROOT', exchange.position.get('INTARIAN_PEPPER_ROOT', 0))
        for o in orders:
            o.symbol = 'INTARIAN_PEPPER_ROOT'

        exchange.execute_orders(orders, order_depths)

    return exchange.get_final_value(intarian_data)


def main():
    datasets_dir = '/Users/minimx/Downloads'
    datasets = []

    for folder_name in os.listdir(datasets_dir):
        folder_path = os.path.join(datasets_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if not folder_name.isdigit() or len(folder_name) != 6:
            continue
        json_file = os.path.join(folder_path, f'{folder_name}.json')
        if not os.path.exists(json_file):
            continue
        datasets.append((folder_name, json_file))

    all_data = []
    for folder_name, json_file in sorted(datasets):
        intarian_data, _ = load_simulator_data(json_file)
        if intarian_data:
            all_data.append((folder_name, intarian_data))

    print("测试首批仓位大小")
    print("=" * 80)
    print(f"{'First':<8} {'Add':<8} {'ProfitTake':<12} {'AvgPnL':<12}")
    print("-" * 80)

    best_pnl = -float('inf')
    best_config = None

    for first_pos in [10, 20, 30, 40, 50]:
        for add_pos in [10, 20]:
            for profit_take in [0.015, 0.020, 0.025]:
                pnls = []
                for folder_name, intarian_data in all_data:
                    pnl = run_strategy(intarian_data, first_pos, add_pos, profit_take, 0.015, 0.025, 20)
                    pnls.append(pnl)
                avg_pnl = sum(pnls) / len(pnls)

                marker = " *" if avg_pnl > best_pnl else ""
                if avg_pnl >= best_pnl:
                    best_pnl = avg_pnl
                    best_config = (first_pos, add_pos, profit_take)
                print(f"{first_pos:<8} {add_pos:<8} {profit_take*100:<10.1f}% {avg_pnl:<12.2f}{marker}")

    print("-" * 80)
    print(f"\n最优配置: First={best_config[0]}, Add={best_config[1]}, ProfitTake={best_config[2]*100:.1f}%")
    print(f"最优AvgPnL: {best_pnl:.2f}")


if __name__ == '__main__':
    main()