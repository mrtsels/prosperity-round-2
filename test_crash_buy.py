"""
测试价格塌陷买入策略：当价格突然下跌时买入，等待反弹
"""

import json
import csv
import os
import sys
from io import StringIO
from typing import List, Dict

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
    for row in reader:
        if row['product'] == 'INTARIAN_PEPPER_ROOT' and row.get('mid_price'):
            mp = float(row['mid_price'])
            if mp > 0:
                intarian_data.append({
                    'timestamp': int(row['timestamp']),
                    'mid_price': mp,
                })
    return intarian_data, []


def create_order_depth(data_row: dict) -> OrderDepth:
    od = OrderDepth()
    od.buy_orders[int(data_row.get('bid_price_1', 13000))] = 10
    od.sell_orders[int(data_row.get('ask_price_1', 13010))] = -10
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


class CrashBuyStrategy:
    """
    价格塌陷买入策略：
    - 正常持仓时，不主动止盈
    - 当价格突然下跌>=crash_threshold时，加仓买入
    - 当价格反弹>=profit_target时，卖出全部仓位
    - 止损：跌破入场价-2%
    """

    def __init__(self, crash_threshold=10, profit_target=5, max_position=80):
        self.position_limit = max_position
        self.crash_threshold = crash_threshold  # 下跌阈值（点）
        self.profit_target = profit_target  # 反弹盈利目标（点）

        self.price_history: Dict[str, List[float]] = {'INTARIAN_PEPPER_ROOT': []}
        self.entry_price: Dict[str, float] = {}
        self.highest_price: Dict[str, float] = {}
        self.consecutive_uptrend = 0
        self.prev_price: Dict[str, float] = {}
        self.has_position = False

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

        # 计算价格变动
        prev = self.prev_price.get(product, mid_price)
        price_change = mid_price - prev
        self.prev_price[product] = mid_price

        history = self.price_history.get(product, [])
        if len(history) < 5:
            return orders

        short_ma = sum(history[-5:]) / 5
        long_ma = sum(history[-10:]) / 10 if len(history) >= 10 else short_ma
        is_uptrend = short_ma > long_ma

        if is_uptrend:
            self.consecutive_uptrend += 1
        else:
            self.consecutive_uptrend = 0

        available = self.position_limit - position

        if position == 0:
            # 入场：MA5>MA10
            if is_uptrend and self.consecutive_uptrend >= 3:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, 20)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))
                    self.entry_price[product] = mid_price
                    self.highest_price[product] = mid_price
                    self.has_position = True

        elif 0 < position < self.position_limit:
            # 加仓：趋势延续+突破
            lookback_high = max(history[-20:]) if len(history) >= 20 else max(history)
            is_breakout = mid_price > lookback_high * 0.999
            if is_uptrend and is_breakout and self.consecutive_uptrend >= 5:
                best_ask = min(od.sell_orders.keys())
                volume = min(available, 20)
                if volume > 0:
                    orders.append(Order(product, int(best_ask), volume))

        elif position > 0:
            if mid_price > self.highest_price[product]:
                self.highest_price[product] = mid_price

            entry = self.entry_price.get(product, mid_price)
            peak = self.highest_price[product]

            # 止盈：价格从低点反弹超过profit_target
            if peak - mid_price <= self.profit_target and (peak - entry) >= self.profit_target:
                # 价格已经在高位，且开始小幅回落，卖出
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
                self.has_position = False
                return orders

            # 止损：跌破入场价-2%
            if mid_price < entry * 0.98:
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
                self.has_position = False
                return orders

            # 移动止损：从高点回落超过2.5%时卖出
            trailing_stop = peak * 0.975
            if mid_price < trailing_stop:
                orders.append(Order(product, int(mid_price), -position))
                self.entry_price[product] = 0
                self.consecutive_uptrend = 0
                self.has_position = False

        return orders


def run_strategy(intarian_data, crash_threshold, profit_target):
    exchange = SimulatedExchange()
    exchange.reset()
    strategy = CrashBuyStrategy(crash_threshold, profit_target)

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

        orders = strategy.signal(state, 'INTARIAN_PEPPER_ROOT', exchange.position.get('INTARIAN_PEPPER_ROOT', 0))
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

    print("价格塌陷买入策略测试")
    print("=" * 70)
    print(f"{'Crash':<8} {'Profit':<8} {'AvgPnL':<12}")
    print("-" * 70)

    best_pnl = -float('inf')
    best_config = None

    for crash in [5, 8, 10, 12, 15]:
        for profit in [3, 5, 8, 10]:
            pnls = []
            for folder_name, intarian_data in all_data:
                pnl = run_strategy(intarian_data, crash, profit)
                pnls.append(pnl)
            avg_pnl = sum(pnls) / len(pnls)

            if avg_pnl > best_pnl:
                best_pnl = avg_pnl
                best_config = (crash, profit)
                print(f"{crash:<8} {profit:<8} {avg_pnl:<12.2f} *")

    print("-" * 70)
    if best_config:
        print(f"\n最优配置: Crash={best_config[0]}, Profit={best_config[1]}")
        print(f"最优AvgPnL: {best_pnl:.2f}")


if __name__ == '__main__':
    main()