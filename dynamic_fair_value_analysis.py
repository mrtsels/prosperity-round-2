"""
动态Fair Value做市策略分析
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
                })
    return ash_data


def calculate_dynamic_fair_value(prices: List[float], window: int = 20) -> List[float]:
    """计算动态fair value（移动平均）"""
    fair_values = []
    for i in range(len(prices)):
        if i < window - 1:
            fair_values.append(sum(prices[:i+1]) / (i+1))
        else:
            fair_values.append(sum(prices[i-window+1:i+1]) / window)
    return fair_values


def analyze_dynamic_mm(data: List[dict], threshold_pct: float = 0.0005) -> dict:
    """
    用动态fair value分析做市策略
    threshold_pct: 价格偏离fair value多少比例时触发交易（如0.0005=0.05%）
    """
    prices = [d['mid_price'] for d in data]
    fair_values = calculate_dynamic_fair_value(prices, window=20)

    # 统计在fair value两侧的分布
    above_fair = 0
    below_fair = 0
    total_deviation = 0

    for i, (price, fv) in enumerate(zip(prices, fair_values)):
        deviation = (price - fv) / fv
        total_deviation += abs(deviation)

        if price > fv:
            above_fair += 1
        elif price < fv:
            below_fair += 1

    n = len(prices)
    avg_deviation_pct = (total_deviation / n) * 100

    # 模拟做市策略
    # 策略：价格低于fair_value×(1-threshold)时买入，高于fair_value×(1+threshold)时卖出
    position = 0
    cash = 0
    trades = 0
    buy_signals = 0
    sell_signals = 0

    threshold_abs = threshold_pct * 10000  # 转换为点数

    for i, d in enumerate(data):
        price = d['mid_price']
        fv = fair_values[i]

        buy_line = fv - threshold_abs
        sell_line = fv + threshold_abs

        # 买入信号
        if price <= buy_line and position <= 0:
            position += 10
            cash -= price * 10
            trades += 1
            buy_signals += 1

        # 卖出信号
        elif price >= sell_line and position >= 0:
            position -= 10
            cash += price * 10
            trades += 1
            sell_signals += 1

        # 平仓信号（价格回到fair value附近）
        elif abs(price - fv) < threshold_abs / 2 and position != 0:
            if position > 0:
                cash += price * position
                position = 0
                trades += 1
            elif position < 0:
                cash += price * abs(position)
                position = 0
                trades += 1

    # 按最后价格平仓
    final_price = prices[-1]
    if position > 0:
        cash += final_price * position
    elif position < 0:
        cash += final_price * abs(position)

    return {
        'n': n,
        'above_fair_pct': above_fair / n * 100,
        'below_fair_pct': below_fair / n * 100,
        'avg_deviation_pct': avg_deviation_pct,
        'trades': trades,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'final_pnl': cash
    }


def main():
    print("=" * 70)
    print("ASH 动态Fair Value做市策略分析")
    print("=" * 70)

    results = []

    for folder_name, json_file in sorted(datasets):
        ash_data = load_ash_data(json_file)
        if not ash_data:
            continue

        print(f"\n数据集: {folder_name}")
        print("-" * 50)

        for threshold in [0.0003, 0.0005, 0.001, 0.002]:
            result = analyze_dynamic_mm(ash_data, threshold)

            print(f"\n阈值: {threshold*100:.2f}% ({threshold*10000:.1f}点)")
            print(f"  价格在fair value上方: {result['above_fair_pct']:.1f}%")
            print(f"  价格在fair value下方: {result['below_fair_pct']:.1f}%")
            print(f"  平均偏离度: {result['avg_deviation_pct']:.3f}%")
            print(f"  买入信号: {result['buy_signals']}")
            print(f"  卖出信号: {result['sell_signals']}")
            print(f"  总交易次数: {result['trades']}")
            print(f"  模拟PnL: {result['final_pnl']:.0f}")

            results.append({
                'dataset': folder_name,
                'threshold': threshold,
                **result
            })

    # 汇总
    print("\n" + "=" * 70)
    print("不同阈值下的交易频率")
    print("=" * 70)

    for threshold in [0.0003, 0.0005, 0.001, 0.002]:
        th_results = [r for r in results if r['threshold'] == threshold]
        if th_results:
            avg_trades = sum(r['trades'] for r in th_results) / len(th_results)
            avg_pnl = sum(r['final_pnl'] for r in th_results) / len(th_results)
            print(f"阈值{threshold*100:.2f}%: 平均交易{avg_trades:.0f}次, 平均PnL={avg_pnl:.0f}")


if __name__ == '__main__':
    main()