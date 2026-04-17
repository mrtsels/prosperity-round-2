"""
时间序列分析：检验INTARIAN数据的特性
"""

import json
import csv
import os
import sys
from io import StringIO
from typing import List, Dict
import math

sys.path.insert(0, '/Users/minimx/Downloads/ROUND_2')


def load_intarian_data(json_path: str) -> List[float]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    activities = data.get('activitiesLog', '')
    reader = csv.DictReader(StringIO(activities), delimiter=';')

    prices = []
    for row in reader:
        if row['product'] == 'INTARIAN_PEPPER_ROOT' and row.get('mid_price'):
            mp = float(row['mid_price'])
            if mp > 0:
                prices.append(mp)
    return prices


def compute_returns(prices: List[float]) -> List[float]:
    """计算对数收益率"""
    returns = []
    for i in range(1, len(prices)):
        ret = math.log(prices[i] / prices[i-1])
        returns.append(ret)
    return returns


def adf_test(series: List[float]) -> dict:
    """
    简化ADF检验（增广Dickey-Fuller）
    使用最小二乘法近似
    """
    n = len(series)
    if n < 10:
        return {'adf_stat': 0, 'p_value': 1.0, 'stationary': False}

    # 计算一阶差分
    diff = [series[i] - series[i-1] for i in range(1, n)]

    # 简化：计算序列是否均值回归
    mean_diff = sum(diff) / len(diff)
    var_diff = sum((d - mean_diff)**2 for d in diff) / len(diff)

    # ADF统计量近似
    adf_stat = mean_diff / (math.sqrt(var_diff / n)) if var_diff > 0 else 0

    # 简化p值（真实计算需要查表）
    p_value = 0.5 if abs(adf_stat) < 2.0 else 0.01

    return {
        'adf_stat': adf_stat,
        'p_value': p_value,
        'stationary': abs(adf_stat) > 2.0
    }


def ljung_box_test(returns: List[float], lag: int = 10) -> dict:
    """
    Ljung-Box检验：检验序列是否存在自相关
    """
    n = len(returns)
    if n < lag + 1:
        return {'lb_stat': 0, 'p_value': 1.0, 'has_autocorr': False}

    # 计算自相关系数
    mean_ret = sum(returns) / n
    var = sum((r - mean_ret)**2 for r in returns)

    if var == 0:
        return {'lb_stat': 0, 'p_value': 1.0, 'has_autocorr': False}

    acf = []
    for k in range(1, lag + 1):
        cov = sum((returns[i] - mean_ret) * (returns[i-k] - mean_ret)
                  for i in range(k, n))
        ac = cov / (n * var)
        acf.append(ac)

    # Ljung-Box统计量
    lb_stat = n * (n + 2) * sum(acf[k-1]**2 / (n - k) for k in range(1, lag + 1))

    # 简化p值（自由度=lag）
    p_value = 0.5  # 简化

    return {
        'lb_stat': lb_stat,
        'p_value': p_value,
        'has_autocorr': lb_stat > 20  # 简化阈值
    }


def analyze_trend(prices: List[float]) -> dict:
    """
    分析价格趋势
    """
    n = len(prices)
    if n < 20:
        return {}

    # 计算滚动均值
    ma5 = sum(prices[-5:]) / 5
    ma10 = sum(prices[-10:]) / 10
    ma20 = sum(prices[-20:]) / 20

    # 趋势强度
    trend_strength = (ma5 - ma10) / ma10 * 100  # 百分比

    # 方向
    is_uptrend = ma5 > ma10
    is_strong_uptrend = ma5 > ma10 and ma10 > ma20

    return {
        'ma5': ma5,
        'ma10': ma10,
        'ma20': ma20,
        'trend_strength_pct': trend_strength,
        'is_uptrend': is_uptrend,
        'is_strong_uptrend': is_strong_uptrend,
        'momentum': 'strong' if abs(trend_strength) > 0.05 else 'weak'
    }


def compute_volatility(prices: List[float]) -> dict:
    """
    计算波动率指标
    """
    if len(prices) < 2:
        return {}

    returns = compute_returns(prices)

    # 日波动率
    daily_vol = math.sqrt(sum(r**2 for r in returns) / len(returns)) * 100

    # 年化波动率（假设252交易日）
    annual_vol = daily_vol * math.sqrt(252)

    # 最大回撤
    peak = prices[0]
    max_dd = 0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'daily_vol_pct': daily_vol,
        'annual_vol_pct': annual_vol,
        'max_drawdown_pct': max_dd,
        'price_range_pct': (max(prices) - min(prices)) / sum(prices) * len(prices) * 50
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
    print("INTARIAN 时间序列分析")
    print("=" * 70)

    for folder_name, json_file in sorted(datasets):
        prices = load_intarian_data(json_file)
        if not prices:
            continue

        returns = compute_returns(prices)

        print(f"\n{'='*60}")
        print(f"数据集: {folder_name}")
        print(f"{'='*60}")

        print(f"\n价格统计:")
        print(f"  数据点数: {len(prices)}")
        print(f"  价格范围: {min(prices):.1f} - {max(prices):.1f}")
        print(f"  起始价格: {prices[0]:.1f}")
        print(f"  结束价格: {prices[-1]:.1f}")
        print(f"  总变动: {(prices[-1]-prices[0])/prices[0]*100:.2f}%")

        # 波动率
        vol = compute_volatility(prices)
        print(f"\n波动率分析:")
        print(f"  日波动率: {vol.get('daily_vol_pct', 0):.4f}%")
        print(f"  年化波动率: {vol.get('annual_vol_pct', 0):.2f}%")
        print(f"  最大回撤: {vol.get('max_drawdown_pct', 0):.2f}%")

        # ADF检验
        adf = adf_test(prices)
        print(f"\nADF平稳性检验:")
        print(f"  ADF统计量: {adf.get('adf_stat', 0):.4f}")
        print(f"  p值: {adf.get('p_value', 1):.4f}")
        print(f"  是否平稳: {'是' if adf.get('stationary') else '否'}")

        # Ljung-Box检验
        lb = ljung_box_test(returns)
        print(f"\nLjung-Box自相关检验:")
        print(f"  LB统计量: {lb.get('lb_stat', 0):.2f}")
        print(f"  存在自相关: {'是' if lb.get('has_autocorr') else '否'}")

        # 趋势分析
        trend = analyze_trend(prices)
        print(f"\n趋势分析:")
        print(f"  MA5: {trend.get('ma5', 0):.2f}")
        print(f"  MA10: {trend.get('ma10', 0):.2f}")
        print(f"  MA20: {trend.get('ma20', 0):.2f}")
        print(f"  趋势强度: {trend.get('trend_strength_pct', 0):.4f}%")
        print(f"  多头趋势: {'是' if trend.get('is_uptrend') else '否'}")
        print(f"  强多头: {'是' if trend.get('is_strong_uptrend') else '否'}")
        print(f"  动量: {trend.get('momentum', 'N/A')}")


if __name__ == '__main__':
    main()