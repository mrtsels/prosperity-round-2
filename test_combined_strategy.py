"""
测试INTARIAN趋势策略 + ASH波段策略组合
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


def load_simulator_data(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)

    activities = data.get('activitiesLog', '')
    reader = csv.DictReader(StringIO(activities), delimiter=';')

    intarian_data = []
    ash_data = []
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
        elif row['product'] == 'ASH_COATED_OSMIUM' and row.get('mid_price'):
            mp = float(row['mid_price'])
            if mp > 0:
                ash_data.append({
                    'timestamp': int(row['timestamp']),
                    'mid_price': mp,
                    'bid_price_1': row.get('bid_price_1', ''),
                    'bid_volume_1': row.get('bid_volume_1', ''),
                    'ask_price_1': row.get('ask_price_1', ''),
                    'ask_volume_1': row.get('ask_volume_1', ''),
                })
    return intarian_data, ash_data


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
        self.position: Dict[str, int] = {'ASH_COATED_OSMIUM': 0, 'INTARIAN_PEPPER_ROOT': 0}
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

    def get_final_value(self, intarian_data, ash_data):
        value = self.cash
        for product, pos in self.position.items():
            if pos != 0:
                if product == 'INTARIAN_PEPPER_ROOT' and intarian_data:
                    value += pos * intarian_data[-1]['mid_price']
                elif product == 'ASH_COATED_OSMIUM' and ash_data:
                    value += pos * ash_data[-1]['mid_price']
        return value


class MomentumStrategy:
    def __init__(self):
        self.position_limit = 80
        self.price_history: Dict[str, List[float]] = {'INTARIAN_PEPPER_ROOT': []}
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
        lookback_high = max(history[-20:]) if len(history) >= 20 else max(history)
        is_breakout = mid_price > lookback_high * 0.999

        if is_uptrend:
            self.consecutive_uptrend += 1
        else:
            self.consecutive_uptrend = 0

        available = self.position_limit - position

        if position == 0:
            if is_uptrend and self.consecutive_uptrend >= 3:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, 20)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    self.entry_price[product] = mid_price
                    self.highest_price[product] = mid_price
        elif 0 < position < self.position_limit:
            if is_uptrend and is_breakout and self.consecutive_uptrend >= 5:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, 20)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
        elif position > 0:
            if mid_price > self.highest_price[product]:
                self.highest_price[product] = mid_price
            entry = self.entry_price[product]
            peak = self.highest_price[product]
            if mid_price < entry * 0.985:
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
            else:
                trailing_stop = peak * 0.975
                if mid_price < trailing_stop:
                    orders.append(Order(product, int(mid_price), -position))
                    self.entry_price[product] = 0
                    self.consecutive_uptrend = 0
        return orders


class ASHSwingStrategy:
    """ASH波段策略 - 围绕10000中心价做均值回归"""

    def __init__(self, center=10000, buy_offset=5, sell_offset=5, stop_offset=15):
        self.center = center
        self.buy_zone = center - buy_offset
        self.sell_zone = center + sell_offset
        self.stop_loss = center - stop_offset
        self.max_position = 30  # ASH仓位上限

    def signal(self, state: TradingState, position: int) -> List[Order]:
        orders = []
        od = state.order_depths.get('ASH_COATED_OSMIUM')
        if od is None or not od.buy_orders or not od.sell_orders:
            return orders

        mid_price = get_mid_price(od)
        if mid_price <= 0:
            return orders

        best_bid, best_ask = get_best_bid_ask(od)

        if position == 0:
            if mid_price <= self.buy_zone:
                volume = min(self.max_position, 20)
                orders.append(Order('ASH_COATED_OSMIUM', int(best_ask), volume))
        elif position > 0:
            if mid_price >= self.sell_zone:
                orders.append(Order('ASH_COATED_OSMIUM', int(best_bid), -position))
            elif mid_price <= self.stop_loss:
                orders.append(Order('ASH_COATED_OSMIUM', int(best_bid), -position))
        return orders


def run_combined_strategy(intarian_data, ash_data, use_ash=True):
    exchange = SimulatedExchange()
    exchange.reset()
    momentum = MomentumStrategy()
    ash_strategy = ASHSwingStrategy() if use_ash else None

    all_timestamps = sorted(set(
        int(r['timestamp']) for r in intarian_data + ash_data
    ))

    for timestamp in all_timestamps:
        intarian_row = next((r for r in intarian_data if int(r['timestamp']) == timestamp), None)
        ash_row = next((r for r in ash_data if int(r['timestamp']) == timestamp), None)

        order_depths = {}
        if intarian_row:
            order_depths['INTARIAN_PEPPER_ROOT'] = create_order_depth(intarian_row)
        if ash_row:
            order_depths['ASH_COATED_OSMIUM'] = create_order_depth(ash_row)

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

        if use_ash and ash_strategy:
            ash_orders = ash_strategy.signal(state, exchange.position.get('ASH_COATED_OSMIUM', 0))
            orders.extend(ash_orders)

        exchange.execute_orders(orders, order_depths)

    return exchange.get_final_value(intarian_data, ash_data)


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
        intarian_data, ash_data = load_simulator_data(json_file)
        if intarian_data and ash_data:
            all_data.append((folder_name, intarian_data, ash_data))

    print("INTARIAN趋势策略 + ASH波段策略 组合测试")
    print("=" * 60)

    print("\n只用INTARIAN趋势:")
    pnls_intarian_only = []
    for folder_name, intarian_data, ash_data in all_data:
        pnl = run_combined_strategy(intarian_data, ash_data, use_ash=False)
        pnls_intarian_only.append(pnl)
        print(f"  {folder_name}: {pnl:.2f}")
    print(f"  平均: {sum(pnls_intarian_only)/len(pnls_intarian_only):.2f}")

    print("\nINTARIAN趋势 + ASH波段:")
    pnls_combined = []
    for folder_name, intarian_data, ash_data in all_data:
        pnl = run_combined_strategy(intarian_data, ash_data, use_ash=True)
        pnls_combined.append(pnl)
        print(f"  {folder_name}: {pnl:.2f}")
    print(f"  平均: {sum(pnls_combined)/len(pnls_combined):.2f}")


if __name__ == '__main__':
    main()