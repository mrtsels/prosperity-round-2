"""
本地回测框架 test.py
用于迭代优化 trader.py

功能：
1. 读取历史CSV数据模拟交易
2. 调用Trader.run()获取订单
3. 模拟订单执行
4. 记录交易log
5. 统计交易statistics
"""

import csv
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ============== 数据模型（复制自官方datamodel.py用于本地测试）==============

class Order:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self):
        return f"Order({self.symbol}, {self.price}, {self.quantity})"

    def __repr__(self):
        return self.__str__()


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(self, symbol: str, price: int, quantity: int,
                 buyer: str = None, seller: str = None, timestamp: int = 0):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self):
        return f"Trade({self.symbol}, {self.price}, {self.quantity}, {self.buyer}<{self.seller})"

    def __repr__(self):
        return self.__str__()


class Listing:
    def __init__(self, symbol: str, product: str, denomination: str):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class Observation:
    def __init__(self):
        self.plainValueObservations: Dict = {}
        self.conversionObservations: Dict = {}


class TradingState:
    def __init__(self,
                 traderData: str,
                 timestamp: int,
                 listings: Dict,
                 order_depths: Dict[str, OrderDepth],
                 own_trades: Dict[str, List[Trade]],
                 market_trades: Dict[str, List[Trade]],
                 position: Dict[str, int],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations


# ============== 历史数据加载器 ==============

class HistoryLoader:
    """加载历史CSV数据"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.products = ['ASH_COATED_OSMIUM', 'INTARIAN_PEPPER_ROOT']

    def load_day(self, day: int) -> Dict[str, List[dict]]:
        """
        加载某一天的数据
        返回: {product: [row1, row2, ...]}
        """
        filename = f"prices_round_2_day_{day}.csv"
        filepath = os.path.join(self.data_dir, filename)

        data = defaultdict(list)
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                product = row['product']
                data[product].append(row)

        # 按timestamp排序
        for product in data:
            data[product].sort(key=lambda x: int(x['timestamp']))

        return dict(data)

    def load_all_days(self) -> Dict[int, Dict[str, List[dict]]]:
        """加载所有可用天数"""
        days = [-1, 0, 1]
        return {day: self.load_day(day) for day in days}


# ============== 模拟交易引擎 ==============

@dataclass
class TradeRecord:
    """单笔交易记录"""
    timestamp: int
    product: str
    direction: str  # 'BUY' or 'SELL'
    price: int
    quantity: int
    pnl_after: float  # 交易后的累计PnL


@dataclass
class PositionSnapshot:
    """持仓快照"""
    timestamp: int
    position: Dict[str, int]
    mid_price: Dict[str, float]
    unrealized_pnl: float


class SimulatedExchange:
    """
    模拟交易所
    负责订单执行和持仓管理
    """

    def __init__(self, position_limit: int = 80):
        self.position_limit = position_limit
        self.position: Dict[str, int] = {'ASH_COATED_OSMIUM': 0, 'INTARIAN_PEPPER_ROOT': 0}
        self.cash: float = 0.0
        self.cost: float = 0.0  # 累计交易成本

        # 历史记录
        self.trade_log: List[TradeRecord] = []
        self.position_snapshots: List[PositionSnapshot] = []
        self.order_log: List[dict] = []  # 记录发送的订单

    def reset(self):
        """重置交易状态"""
        self.position = {p: 0 for p in self.position}
        self.cash = 0.0
        self.cost = 0.0
        self.trade_log = []
        self.position_snapshots = []
        self.order_log = []

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """计算中间价"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def execute_orders(self,
                      orders: List[Order],
                      order_depths: Dict[str, OrderDepth],
                      timestamp: int) -> Tuple[float, float]:
        """
        执行订单
        返回: (realized_pnl, cost)
        """
        realized_pnl = 0.0
        cost = 0.0

        for order in orders:
            product = order.symbol
            od = order_depths.get(product)
            if od is None:
                continue

            position = self.position[product]
            qty = order.quantity  # 正=买入，负=卖出

            if qty > 0:  # 买单 - 与卖单成交
                if od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                    available_qty = abs(od.sell_orders[best_ask])
                    exec_qty = min(qty, available_qty)
                    exec_price = best_ask

                    # 更新持仓
                    self.position[product] += exec_qty
                    self.cash -= exec_price * exec_qty
                    cost += exec_price * exec_qty * 0.0001  # 万分之一手续费

                    # 记录交易
                    self.trade_log.append(TradeRecord(
                        timestamp=timestamp,
                        product=product,
                        direction='BUY',
                        price=exec_price,
                        quantity=exec_qty,
                        pnl_after=self.cash + self.get_portfolio_value(order_depths)
                    ))

                    # 订单log
                    self.order_log.append({
                        'timestamp': timestamp,
                        'product': product,
                        'type': 'BUY',
                        'price': exec_price,
                        'quantity': exec_qty,
                        'exec_price': exec_price
                    })

            elif qty < 0:  # 卖单 - 与买单成交
                if od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    available_qty = od.buy_orders[best_bid]
                    exec_qty = min(abs(qty), available_qty)
                    exec_price = best_bid

                    # 更新持仓（允许做空，但不能超过限制）
                    new_position = self.position[product] - exec_qty
                    if abs(new_position) <= self.position_limit:
                        self.position[product] = new_position
                        self.cash += exec_price * exec_qty
                        cost += exec_price * exec_qty * 0.0001

                        # 记录交易
                        self.trade_log.append(TradeRecord(
                            timestamp=timestamp,
                            product=product,
                            direction='SELL',
                            price=exec_price,
                            quantity=exec_qty,
                            pnl_after=self.cash + self.get_portfolio_value(order_depths)
                        ))

                        # 订单log
                        self.order_log.append({
                            'timestamp': timestamp,
                            'product': product,
                            'type': 'SELL',
                            'price': exec_price,
                            'quantity': exec_qty,
                            'exec_price': exec_price
                        })

        return realized_pnl, cost

    def get_portfolio_value(self, order_depths: Dict[str, OrderDepth]) -> float:
        """计算组合市值"""
        value = self.cash
        for product, pos in self.position.items():
            if pos != 0:
                od = order_depths.get(product)
                if od:
                    mid = self.get_mid_price(od)
                    value += pos * mid
        return value

    def snapshot(self, timestamp: int, order_depths: Dict[str, OrderDepth]):
        """记录持仓快照"""
        mid_prices = {}
        for product, od in order_depths.items():
            mid_prices[product] = self.get_mid_price(od)

        self.position_snapshots.append(PositionSnapshot(
            timestamp=timestamp,
            position=dict(self.position),
            mid_price=mid_prices,
            unrealized_pnl=0.0  # 简化计算
        ))


# ============== 回测框架 ==============

class Backtester:
    """
    回测框架
    """

    def __init__(self, data_dir: str, trader_class):
        self.data_dir = data_dir
        self.trader_class = trader_class
        self.loader = HistoryLoader(data_dir)
        self.exchange = SimulatedExchange(position_limit=80)

        # 统计结果
        self.results: Dict[int, dict] = {}

    def _create_trading_state(self,
                               timestamp: int,
                               order_depths_data: Dict[str, List[dict]],
                               current_idx: int) -> TradingState:
        """从历史数据创建TradingState"""

        listings = {
            'ASH_COATED_OSMIUM': Listing('ASH_COATED_OSMIUM', 'ASH_COATED_OSMIUM', 'XIRECS'),
            'INTARIAN_PEPPER_ROOT': Listing('INTARIAN_PEPPER_ROOT', 'INTARIAN_PEPPER_ROOT', 'XIRECS')
        }

        order_depths = {}
        for product in self.loader.products:
            rows = order_depths_data.get(product, [])
            if current_idx < len(rows):
                row = rows[current_idx]
            else:
                row = rows[-1] if rows else {}

            od = OrderDepth()

            # 解析买单
            for i in range(1, 4):
                price_key = f'bid_price_{i}'
                vol_key = f'bid_volume_{i}'
                if price_key in row and vol_key in row:
                    p = row.get(price_key, '')
                    v = row.get(vol_key, '')
                    if p and v and p != '' and v != '':
                        try:
                            od.buy_orders[int(p)] = int(v)
                        except (ValueError, TypeError):
                            pass

            # 解析卖单
            for i in range(1, 4):
                price_key = f'ask_price_{i}'
                vol_key = f'ask_volume_{i}'
                if price_key in row and vol_key in row:
                    p = row.get(price_key, '')
                    v = row.get(vol_key, '')
                    if p and v and p != '' and v != '':
                        try:
                            od.sell_orders[int(p)] = -int(v)  # 卖单数量为负
                        except (ValueError, TypeError):
                            pass

            order_depths[product] = od

        # 创建空的trades和observations
        own_trades = {p: [] for p in self.loader.products}
        market_trades = {p: [] for p in self.loader.products}
        observations = Observation()

        return TradingState(
            traderData='',
            timestamp=timestamp,
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades,
            market_trades=market_trades,
            position=dict(self.exchange.position),
            observations=observations
        )

    def run_day(self, day: int, verbose: bool = True) -> dict:
        """运行单日回测"""

        # 重置状态
        self.exchange.reset()
        trader = self.trader_class()

        # 加载数据
        all_data = self.loader.load_day(day)
        if not all_data:
            return {}

        # 获取时间戳序列
        timestamps = sorted(set(
            int(row['timestamp'])
            for rows in all_data.values()
            for row in rows
        ))

        if verbose:
            print(f"\n{'='*60}")
            print(f"回测 Day {day}")
            print(f"{'='*60}")
            print(f"时间步数: {len(timestamps)}")

        daily_stats = {
            'day': day,
            'trades': 0,
            'pnl': 0.0,
            'cost': 0.0,
            'final_value': 0.0,
            'order_log': []
        }

        # 逐时间步执行
        for idx, timestamp in enumerate(timestamps):
            # 构建当前order_depths
            order_depths_data = {}
            for product in self.loader.products:
                rows = all_data.get(product, [])
                if idx < len(rows):
                    order_depths_data[product] = rows[:idx+1]

            # 创建TradingState
            state = self._create_trading_state(timestamp, all_data, idx)

            # 调用trader
            try:
                result, conversions, traderData = trader.run(state)
            except Exception as e:
                if verbose:
                    print(f"Error at timestamp {timestamp}: {e}")
                continue

            # 执行订单
            orders_by_product = result if result else {}
            all_orders = []
            for product, orders in orders_by_product.items():
                for order in orders:
                    order.symbol = product  # 确保symbol正确
                    all_orders.append(order)

            # 执行
            pnl_delta, cost = self.exchange.execute_orders(
                all_orders, state.order_depths, timestamp
            )

            daily_stats['trades'] += len(all_orders)
            daily_stats['cost'] += cost
            daily_stats['order_log'].extend(self.exchange.order_log[-len(all_orders):])

            # 记录快照
            self.exchange.snapshot(timestamp, state.order_depths)

        # 计算最终PnL
        final_value = self.exchange.cash
        for product, pos in self.exchange.position.items():
            if pos != 0:
                od = all_data.get(product, [{}])
                if od:
                    last_row = od[-1]
                    mid_price = float(last_row.get('mid_price', 0))
                    final_value += pos * mid_price

        daily_stats['pnl'] = final_value
        daily_stats['final_value'] = final_value

        if verbose:
            print(f"交易次数: {daily_stats['trades']}")
            print(f"交易成本: {daily_stats['cost']:.2f}")
            print(f"最终PnL: {daily_stats['pnl']:.2f}")
            print(f"最终持仓: {dict(self.exchange.position)}")

        return daily_stats

    def run_all(self, verbose: bool = True) -> dict:
        """运行所有天数回测"""

        all_results = {}

        for day in [-1, 0, 1]:
            result = self.run_day(day, verbose=verbose)
            all_results[day] = result

        # 汇总统计
        total_trades = sum(r['trades'] for r in all_results.values())
        total_cost = sum(r['cost'] for r in all_results.values())
        total_pnl = sum(r['pnl'] for r in all_results.values())

        if verbose:
            print(f"\n{'='*60}")
            print("汇总统计")
            print(f"{'='*60}")
            print(f"总交易次数: {total_trades}")
            print(f"总交易成本: {total_cost:.2f}")
            print(f"总PnL: {total_pnl:.2f}")

        self.results = all_results
        return all_results

    def get_statistics(self, verbose: bool = True) -> dict:
        """计算详细统计"""

        if not self.results:
            self.run_all(verbose=False)

        all_trades = []
        for day_result in self.results.values():
            all_trades.extend(day_result.get('order_log', []))

        if not all_trades:
            return {}

        # 基本统计
        stats = {
            'total_trades': len(all_trades),
            'total_pnl': sum(r['pnl'] for r in self.results.values()),
            'total_cost': sum(r['cost'] for r in self.results.values()),
            'by_product': {},
            'by_day': {}
        }

        # 按产品统计
        for product in self.loader.products:
            product_trades = [t for t in all_trades if t['product'] == product]
            if product_trades:
                buys = [t for t in product_trades if t['type'] == 'BUY']
                sells = [t for t in product_trades if t['type'] == 'SELL']
                stats['by_product'][product] = {
                    'total_trades': len(product_trades),
                    'buys': len(buys),
                    'sells': len(sells),
                    'avg_price': sum(t['exec_price'] for t in product_trades) / len(product_trades)
                }

        # 按天统计
        for day, result in self.results.items():
            stats['by_day'][day] = {
                'trades': result['trades'],
                'pnl': result['pnl'],
                'cost': result['cost']
            }

        if verbose:
            print(f"\n{'='*60}")
            print("详细统计")
            print(f"{'='*60}")
            print(f"总交易次数: {stats['total_trades']}")
            print(f"总PnL: {stats['total_pnl']:.2f}")
            print(f"总成本: {stats['total_cost']:.2f}")

            print("\n按产品:")
            for product, pstats in stats['by_product'].items():
                print(f"  {product}: {pstats['total_trades']}笔, 均价{pstats['avg_price']:.2f}")

            print("\n按天:")
            for day, dstats in stats['by_day'].items():
                print(f"  Day {day}: {dstats['trades']}笔, PnL={dstats['pnl']:.2f}")

        return stats


# ============== 主程序 ==============

def run_backtest(trader_class, data_dir: str = None, verbose: bool = True):
    """
    运行回测的便捷函数

    用法:
        from trader import Trader
        from test import run_backtest
        run_backtest(Trader)
    """

    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    backtester = Backtester(data_dir, trader_class)

    # 运行回测
    print("开始回测...")
    print(f"数据目录: {data_dir}")

    results = backtester.run_all(verbose=verbose)
    stats = backtester.get_statistics(verbose=verbose)

    # 生成交易log文件
    log_file = os.path.join(data_dir, 'trade_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'product', 'type', 'price', 'quantity', 'exec_price'])
        writer.writeheader()
        for day_result in results.values():
            for order in day_result.get('order_log', []):
                writer.writerow(order)

    if verbose:
        print(f"\n交易log已保存到: {log_file}")

    return results, stats


if __name__ == '__main__':
    # 示例：导入trader并运行回测
    print("="*60)
    print("本地回测框架")
    print("="*60)
    print("\n使用方法:")
    print("  from test import run_backtest")
    print("  from trader import Trader")
    print("  results, stats = run_backtest(Trader)")
    print()
    print("或直接运行:")
    print("  python test.py")
    print("  (需要创建空的Trader类用于测试)")
