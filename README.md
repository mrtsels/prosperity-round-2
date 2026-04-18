# IMC Prosperity Round 2 交易策略

## 策略概述

双股票量化交易策略，包含:

- **ASH_COATED_OSMIUM**: 稳定均值回归产品，做市策略
- **INTARIAN_PEPPER_ROOT**: 趋势动量产品，趋势跟随策略

**回测结果**: 288,052 PnL (三天 round2 数据)

---

## 策略逻辑

### ASH 做市策略

**核心思想**: ASH 围绕 10000 均值回归，通过 EMA 追踪 Fair Value，被动挂单赚取 spread

**关键参数**:
- `ASH_ANCHOR = 10000`: 锚定价格
- `ASH_POSITION_SKEW = 0.1`: 仓位对 FV 的影响系数
- `ASH_TAKE_EDGE = 0.0`: 不做主动 taker，只做 maker
- `ASH_INNER_WIDTH_WIDE = 1.0`: 宽 spread 时的挂单宽度

**订单生成**:
1. 计算 Fair Value: `fair = 0.75 * 10000 + 0.25 * EMA(mid) - 0.1 * position`
2. 双层挂单：内层紧贴 fair，外层更宽
3. 仓位偏向：多头时增加卖出，空头时增加买入

### INTARIAN 趋势策略

**核心思想**: 趋势跟随，MA 多头排列入场，突破加仓，移动止损

**关键参数**:
- `INT_MA_SHORT = 5`, `INT_MA_LONG = 8`: 均线周期
- `INT_FIRST_SIZE = 20`: 首批入场数量
- `INT_ADD_CONSEC = 1`: 加仓所需的连续趋势周期
- `INT_STOP_LOSS_PCT = 0.015`: 固定止损 1.5%
- `INT_TRAILING_STOP_PCT = 0.025`: 移动止损 2.5%

**订单生成**:
1. 入场：MA5 > MA8 + 连续1周期
2. 加仓：MA5 > MA8 + 突破20日高点 + 连续1周期
3. 止损：跌破入场价1.5% 或 从最高点回撤2.5%

---

## 回测脚本使用

### 首次安装 Rust 环境

```bash
# 1. 安装 Xcode 命令行工具
xcode-select --install

# 2. 安装 Rust
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"

# 3. 验证安装
cargo --version
```

### 运行回测

```bash
# 进入 backtester 目录
cd $HOME/Downloads/prosperity_rust_backtester

# 加载 Rust 环境
source "$HOME/.cargo/env"

# 运行回测（默认测试 round2 数据集）
make backtest TRADER=$HOME/Downloads/ROUND_2/trader.py

# 或使用 rust_backtester 命令
rust_backtester --trader $HOME/Downloads/ROUND_2/trader.py --dataset round2
```

### 可用命令

```bash
# 测试不同数据集
make tutorial                    # 测试 tutorial 数据
make round2 TRADER=/path/to/trader.py   # 测试 round2 数据

# 指定某一天
make backtest TRADER=/path/to/trader.py DAY=-1

# 完整持久化（生成详细日志）
make backtest TRADER=/path/to/trader.py PERSIST=1
```

### 输出示例

```
trader: trader.py
dataset: round2 [default]
mode: fast
artifacts: log-only
SET             DAY    TICKS  OWN_TRADES    FINAL_PNL  RUN_DIR
D-1              -1    10000         724     96001.00  runs/backtest-xxx-round2-day-1
D=0               0    10000         710     96000.00  runs/backtest-xxx-round2-day-0
D+1               1    10000         719     96051.00  runs/backtest-xxx-round2-day-1
TOTAL             -    30000        2153    288052.00  -

PRODUCT                     D-1        D=0        D+1      TOTAL
INTARIAN_PEPPER_ROOT   79311.00   79273.00   79172.00  237756.00
ASH_COATED_OSMIUM      16690.00   16727.00   16879.00   50296.00
```

---

## 文件结构

```
ROUND_2/
├── trader.py           # 策略代码
├── datamodel.py        # 数据模型定义
└── README.md           # 本文件
```

---

## 回测器目录

```
prosperity_rust_backtester/
├── README.md           # backtester 使用说明
├── datasets/
│   ├── tutorial/       # 教程数据
│   ├── round1/        # Round 1 数据
│   ├── round2/        # Round 2 数据
│   └── ...
├── traders/           #  bundled traders
└── src/              # Rust 源代码
```
