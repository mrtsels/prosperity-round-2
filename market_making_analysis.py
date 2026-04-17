"""
高频做市策略可用性分析
"""

import json
import csv
import os
from io import StringIO
from typing import List, Dict, Tuple

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


def load_ash_data(json_path: str) -> Tuple[List[dict], List[dict]]:
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


def analyze_market_making(data: List[dict], fair_value: float = 10000) -> dict:
    """
    分析做市策略的可行性
    """
    if not data:
        return {}

    mid_prices = [d['mid_price'] for d in data]

    # 价格统计
    mean_price = sum(mid_prices) / len(mid_prices)
    min_price = min(mid_prices)
    max_price = max(mid_prices)

    # 分布在fair_value两侧的比例
    above_fair = sum(1 for p in mid_prices if p > fair_value)
    below_fair = sum(1 for p in mid_prices if p < fair_value)
    at_fair = sum(1 for p in mid_prices if p == fair_value)

    # 偏离fair_value的绝对值
    deviations = [abs(p - fair_value) for p in mid_prices]
    avg_deviation = sum(deviations) / len(deviations)

    # 计算理论上可赚取的价差
    # 做市策略：买入价 = fair - spread/2, 卖出价 = fair + spread/2
    spread = 1  # 假设1点spread
    profit_per_round_trip = spread

    # 统计价格变动
    price_changes = []
    for i in range(1, len(data)):
        change = data[i]['mid_price'] - data[i-1]['mid_price']
        price_changes.append(change)

    # 计算每笔交易的期望收益
    # 如果价格从10000跌到9995再涨回10000，买入后持仓不动会亏5点
    # 如果价格从10000涨到10005再跌回10000，卖出后空仓会少赚5点

    # 模拟简单的做市策略
    position = 0  # 0=空仓, 正=多头, 负=空头
    trades = 0
    cash = 0

    for i, d in enumerate(data):
        mid = d['mid_price']

        # 简单的均值回归策略
        if mid < fair_value - 3 and position <= 0:  # 价格低于fair value 3点，买入
            # 假设能以mid+1的价格买入（付1点spread）
            position += 10  # 买入10股
            cash -= (mid + 1) * 10
            trades += 1
        elif mid > fair_value + 3 and position >= 0:  # 价格高于fair value 3点，卖出
            position -= 10  # 卖出10股
            cash += (mid - 1) * 10  # 以mid-1卖出（收1点spread）
            trades += 1
        elif mid >= fair_value - 1 and mid <= fair_value + 1 and position != 0:
            # 价格回到fair value附近，平仓
            if position > 0:
                cash += (mid - 1) * position
                trades += 1
            else:
                cash += (mid + 1) * abs(position)
                trades += 1
            position = 0

    # 最终按最后价格平仓
    if position != 0:
        final_price = data[-1]['mid_price']
        if position > 0:
            cash += (final_price - 1) * position
        else:
            cash += (final_price + 1) * abs(position)

    return {
        'data_points': len(data),
        'mean_price': mean_price,
        'price_range': (min_price, max_price),
        'above_fair_pct': above_fair / len(data) * 100,
        'below_fair_pct': below_fair / len(data) * 100,
        'avg_deviation': avg_deviation,
        'final_cash': cash,
        'trades': trades
    }


def main():
    print("=" * 70)
    print("ASH 高频做市策略可用性分析")
    print("=" * 70)

    for folder_name, json_file in sorted(datasets):
        ash_data = load_ash_data(json_file)
        if not ash_data:
            continue

        result = analyze_market_making(ash_data)

        print(f"\n数据集: {folder_name}")
        print("-" * 50)
        print(f"数据点数: {result['data_points']}")
        print(f"价格均值: {result['mean_price']:.2f}")
        print(f"价格范围: {result['price_range'][0]:.1f} - {result['price_range'][1]:.1f}")
        print(f"价格在fair value以上: {result['above_fair_pct']:.1f}%")
        print(f"价格在fair value以下: {result['below_fair_pct']:.1f}%")
        print(f"平均偏离度: {result['avg_deviation']:.2f}")
        print(f"模拟交易次数: {result['trades']}")
        print(f"模拟收益: {result['final_cash']:.2f}")

    # 汇总分析
    print("\n" + "=" * 70)
    print("汇总分析")
    print("=" * 70)

    print("""
关键发现：
1. ASH价格范围 9989-10017，fair value=10000
2. 价格在10000上方和下方的时间比例接近50/50
3. 平均偏离度约3-5点

做市策略可行性：
- 如果价格围绕10000对称波动，适合均值回归做市
- 关键问题：MAF限制了交易频率，不能乱挂单
- 需要在价格高度偏离fair value时才挂单

策略建议：
- 买入阈值：价格 < fair_value - 5（偏离5点以上）
- 卖出阈值：价格 > fair_value + 5（偏离5点以上）
- 止损：价格继续偏离超过10点
""")


if __name__ == '__main__':
    main()