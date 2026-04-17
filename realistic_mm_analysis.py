"""
考虑Spread和MAF成本的真实做市策略模拟
"""

import json
import csv
import os
from io import StringIO
from typing import List, Dict

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


def calculate_fair_value(prices: List[float], window: int = 20) -> List[float]:
    """计算动态fair value"""
    fv = []
    for i in range(len(prices)):
        if i < window:
            fv.append(sum(prices[:i+1]) / (i+1))
        else:
            fv.append(sum(prices[i-window+1:i+1]) / window)
    return fv


def simulate_market_making(data: List[dict], threshold: float, max_position: int = 30,
                           spread_cost: float = 1.0, maf_cost: float = 0,
                           maf_limit: int = None) -> dict:
    """
    模拟做市策略

    threshold: 偏离fair value多少时触发交易（绝对价格）
    max_position: 最大持仓
    spread_cost: 每次交易的成本（点）
    maf_cost: MAF总成本（如果交易次数超限）
    maf_limit: MAF允许的最大交易次数
    """
    prices = [d['mid_price'] for d in data]
    fair_values = calculate_fair_value(prices)

    position = 0  # 正=多头，负=空头
    cash = 0
    trades = 0
    buy_trades = 0
    sell_trades = 0
    rejected = 0

    for i, d in enumerate(data):
        price = d['mid_price']
        fv = fair_values[i]

        # 检查MAF限制
        if maf_limit and trades >= maf_limit:
            rejected += 1
            continue

        buy_line = fv - threshold
        sell_line = fv + threshold

        # 买入逻辑
        if price <= buy_line and position < max_position:
            # 买单 - 用市场价成交（付spread）
            exec_price = price + spread_cost
            volume = min(10, max_position - position)
            position += volume
            cash -= exec_price * volume
            trades += 1
            buy_trades += 1

        # 卖出逻辑
        elif price >= sell_line and position > -max_position:
            # 卖单 - 用市场价成交（收spread）
            exec_price = price - spread_cost
            volume = min(10, max_position + position)
            position -= volume
            cash += exec_price * volume
            trades += 1
            sell_trades += 1

        # 平仓逻辑（价格回到fair value附近）
        elif abs(price - fv) < threshold / 2:
            if position > 0:
                exec_price = price - spread_cost
                cash += exec_price * position
                trades += 1
                position = 0
            elif position < 0:
                exec_price = price + spread_cost
                cash += exec_price * abs(position)
                trades += 1
                position = 0

    # 最终平仓
    if position != 0:
        final_price = prices[-1]
        if position > 0:
            cash += (final_price - spread_cost) * position
        else:
            cash += (final_price + spread_cost) * abs(position)
        trades += 1

    # 扣除MAF成本
    net_cash = cash - maf_cost

    return {
        'trades': trades,
        'buy_trades': buy_trades,
        'sell_trades': sell_trades,
        'rejected': rejected,
        'final_position': position,
        'gross_pnl': cash,
        'maf_cost': maf_cost,
        'net_pnl': net_cash
    }


def main():
    print("=" * 70)
    print("ASH 做市策略 - 真实成本模拟")
    print("=" * 70)

    results = []

    for folder_name, json_file in sorted(datasets):
        ash_data = load_ash_data(json_file)
        if not ash_data:
            continue

        print(f"\n{'='*60}")
        print(f"数据集: {folder_name}")
        print(f"{'='*60}")

        for threshold in [3, 5, 8, 10]:
            for spread in [1, 2]:
                for maf_limit in [None, 50, 100, 200]:
                    for maf_cost in [0, 100, 500, 1000]:
                        r = simulate_market_making(
                            ash_data, threshold,
                            spread_cost=spread,
                            maf_cost=maf_cost if maf_limit and r['trades'] > maf_limit else 0,
                            maf_limit=maf_limit
                        )
                        r['threshold'] = threshold
                        r['spread'] = spread
                        r['maf_limit'] = maf_limit
                        r['maf_cost'] = maf_cost
                        results.append({'dataset': folder_name, **r})

    # 按阈值分组汇总
    print("\n" + "=" * 70)
    print("汇总：不同阈值的平均表现")
    print("=" * 70)

    for threshold in [3, 5, 8, 10]:
        th_results = [r for r in results if r['threshold'] == threshold]
        if th_results:
            avg_trades = sum(r['trades'] for r in th_results) / len(th_results)
            avg_pnl = sum(r['net_pnl'] for r in th_results) / len(th_results)
            avg_buy = sum(r['buy_trades'] for r in th_results) / len(th_results)
            avg_sell = sum(r['sell_trades'] for r in th_results) / len(th_results)

            print(f"\n阈值={threshold}点:")
            print(f"  平均交易次数: {avg_trades:.0f} (买{avg_buy:.0f}, 卖{avg_sell:.0f})")
            print(f"  平均PnL: {avg_pnl:.0f}")

    # 找出最优配置
    print("\n" + "=" * 70)
    print("最优配置 (不考虑MAF)")
    print("=" * 70)

    best_no_maf = max([r for r in results if r['maf_limit'] is None],
                       key=lambda x: x['net_pnl'])
    print(f"阈值={best_no_maf['threshold']}, spread={best_no_maf['spread']}")
    print(f"  交易次数: {best_no_maf['trades']}")
    print(f"  PnL: {best_no_maf['net_pnl']:.0f}")

    print("\n" + "=" * 70)
    print("最优配置 (考虑MAF限制=100, MAF成本=500)")
    print("=" * 70)

    constrained = [r for r in results
                  if r['maf_limit'] == 100 and r['maf_cost'] == 500]
    if constrained:
        best_constrained = max(constrained, key=lambda x: x['net_pnl'])
        print(f"阈值={best_constrained['threshold']}, spread={best_constrained['spread']}")
        print(f"  交易次数: {best_constrained['trades']} (限制: {best_constrained['maf_limit']})")
        print(f"  被拒绝: {best_constrained['rejected']}")
        print(f"  MAF成本: {best_constrained['maf_cost']}")
        print(f"  PnL: {best_constrained['net_pnl']:.0f}")

    # MAF成本分析
    print("\n" + "=" * 70)
    print("MAF成本影响分析 (阈值=5, spread=1)")
    print("=" * 70)

    maf_analysis = [r for r in results
                    if r['threshold'] == 5 and r['spread'] == 1
                    and r['maf_limit'] == 100]

    for r in sorted(maf_analysis, key=lambda x: x['maf_cost']):
        print(f"MAF成本={r['maf_cost']:>5}: PnL={r['net_pnl']:>12.0f}, 交易={r['trades']}")


if __name__ == '__main__':
    main()