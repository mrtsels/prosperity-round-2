"""
ASH做市策略 - 真实成本模拟（修复版）
"""

import json
import csv
import os
import sys
from io import StringIO
from typing import List, Dict

sys.path.insert(0, '/Users/minimx/Downloads/ROUND_2')
from datamodel import OrderDepth, TradingState, Order


def load_ash_data(json_path: str) -> List[dict]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    activities = data.get('activitiesLog', '')
    reader = csv.DictReader(StringIO(activities), delimiter=';')

    ash_data = []
    for row in reader:
        if row['product'] == 'ASH_COATED_OSMIUM' and row.get('mid_price'):
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
    return ash_data


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


class MarketMaker:
    """
    ASH做市商
    - 挂买单在buy_price
    - 挂卖单在sell_price
    - 当市场价触价时自动成交
    """

    def __init__(self, fair_value: float = 10000, threshold: float = 5,
                 max_position: int = 30, spread: int = 1):
        self.fair_value = fair_value
        self.threshold = threshold  # 偏离多少点时挂单
        self.max_position = max_position
        self.spread = spread  # 买卖价差

        self.buy_price = None  # 挂单买入价
        self.sell_price = None  # 挂单卖出价
        self.position = 0  # 当前持仓
        self.cash = 0  # 现金
        self.trades = 0  # 成交次数

    def compute_orders(self, mid_price: float, buy_orders: dict, sell_orders: dict):
        """计算应该挂什么单"""
        orders = []

        # 计算挂单价格
        bid_price = int(mid_price - self.threshold - self.spread)
        ask_price = int(mid_price + self.threshold + self.spread)

        available_buy = self.max_position - self.position  # 还能买多少
        available_sell = self.max_position + self.position  # 还能卖多少（做空）

        # 买单：价格 <= bid_price 时买入
        if available_buy > 0:
            # 检查当前是否有更好的买单
            if self.buy_price is None or bid_price < self.buy_price:
                # 取消旧买单，挂新买单
                if self.buy_price is not None:
                    orders.append(Order('ASH_COATED_OSMIUM', self.buy_price, -self.position))  # 取消旧单
                orders.append(Order('ASH_COATED_OSMIUM', bid_price, available_buy))
                self.buy_price = bid_price

        # 卖单：价格 >= ask_price 时卖出
        if available_sell > 0:
            if self.sell_price is None or ask_price > self.sell_price:
                if self.sell_price is not None:
                    orders.append(Order('ASH_COATED_OSMIUM', self.sell_price, self.position))  # 取消旧单
                orders.append(Order('ASH_COATED_OSMIUM', ask_price, available_sell))
                self.sell_price = ask_price

        return orders

    def execute_at_market(self, mid_price: float, buy_orders: dict, sell_orders: dict):
        """按市场价成交"""
        # 检查是否被成交
        trades_made = []

        # 买单被卖单成交（有人卖给我们）
        if self.buy_price is not None and self.buy_price in sell_orders:
            qty = min(abs(sell_orders[self.buy_price]), self.max_position - self.position)
            if qty > 0:
                self.position += qty
                self.cash -= self.buy_price * qty
                self.trades += 1
                trades_made.append(('BUY', self.buy_price, qty))

        # 卖单被买单成交（有人买我们的）
        if self.sell_price is not None and self.sell_price in buy_orders:
            qty = min(abs(buy_orders[self.sell_price]), self.max_position + self.position)
            if qty > 0:
                self.position -= qty
                self.cash += self.sell_price * qty
                self.trades += 1
                trades_made.append(('SELL', self.sell_price, qty))

        return trades_made

    def reset(self):
        self.buy_price = None
        self.sell_price = None
        self.position = 0
        self.cash = 0
        self.trades = 0


def run_backtest(data: List[dict], threshold: float, spread: int = 1,
                maf_cost: float = 0, maf_trade_limit: int = None) -> dict:
    """回测做市策略"""

    mm = MarketMaker(threshold=threshold, spread=spread)

    buy_count = 0
    sell_count = 0

    for i, d in enumerate(data):
        mid_price = d['mid_price']

        # 获取当前簿
        od = create_order_depth(d)
        buy_orders = od.buy_orders
        sell_orders = {k: -v for k, v in od.sell_orders.items()}

        # 检查是否被成交
        trades = mm.execute_at_market(mid_price, buy_orders, sell_orders)

        for trade_type, price, qty in trades:
            if trade_type == 'BUY':
                buy_count += 1
            else:
                sell_count += 1

        # 更新挂单
        mm.compute_orders(mid_price, buy_orders, sell_orders)

        # 检查MAF限制
        if maf_trade_limit and mm.trades >= maf_trade_limit:
            break

    # 最终平仓
    final_price = data[-1]['mid_price']
    if mm.position > 0:
        mm.cash += final_price * mm.position
    elif mm.position < 0:
        mm.cash += final_price * abs(mm.position)

    # 扣除MAF成本
    net_pnl = mm.cash - maf_cost

    return {
        'threshold': threshold,
        'spread': spread,
        'trades': mm.trades,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'final_position': mm.position,
        'gross_pnl': mm.cash,
        'maf_cost': maf_cost,
        'net_pnl': net_pnl
    }


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

    print("=" * 70)
    print("ASH做市策略 - 真实成本模拟")
    print("=" * 70)

    all_results = []

    for folder_name, json_file in sorted(datasets):
        ash_data = load_ash_data(json_file)
        if not ash_data:
            continue

        print(f"\n数据集: {folder_name}, 数据点: {len(ash_data)}")

        for threshold in [3, 5, 8]:
            for spread in [1, 2]:
                r = run_backtest(ash_data, threshold, spread)
                r['dataset'] = folder_name
                all_results.append(r)
                print(f"  阈值={threshold}, spread={spread}: "
                      f"交易={r['trades']:>3}, 买={r['buy_count']:>3}, 卖={r['sell_count']:>3}, "
                      f"PnL={r['net_pnl']:>12.0f}")

    # 汇总
    print("\n" + "=" * 70)
    print("汇总（4个数据集平均）")
    print("=" * 70)

    for threshold in [3, 5, 8]:
        for spread in [1, 2]:
            configs = [r for r in all_results if r['threshold'] == threshold and r['spread'] == spread]
            if configs:
                avg_trades = sum(r['trades'] for r in configs) / len(configs)
                avg_pnl = sum(r['net_pnl'] for r in configs) / len(configs)
                print(f"阈值={threshold}, spread={spread}: 平均交易={avg_trades:.0f}, 平均PnL={avg_pnl:.0f}")


if __name__ == '__main__':
    main()