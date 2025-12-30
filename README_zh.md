# Gold Machine

一个基于机器学习的黄金ETF和期货价格预测系统，配备由ATR（平均真实波幅）策略驱动的先进风险管理功能。

## 功能特性

- **机器学习预测**：支持多种算法（LinearRegression、FastTree、FastForest）及集成模型
- **ATR风险管理**：动态止损、止盈和仓位管理
- **Walk-Forward回测**：使用扩展窗口进行真实样本外测试
- **技术指标**：MA、RSI、ATR、MACD、布林带等
- **交互式可视化**：价格预测图表和累计收益分析

## 支持的数据源

> https://akshare.akfamily.xyz/index.html

### 黄金ETF数据（默认）
- API端点：`fund_etf_hist_em`
- 代码：`518880`（GLD ETF）
- 数据字段：日期、开盘、最高、最低、收盘、成交量

### 上海黄金交易所（SGE）期货
- API端点：`spot_hist_sge`
- 代码：`Au99.99`（黄金期货）
- 数据字段：日期、开盘、最高、最低、收盘价格

## 快速开始

```bash
# 使用默认ETF数据（GLD ETF）
dotnet run

# 使用自定义ETF代码和集成模型
dotnet run -- --etf 159831 --ensemble

# 使用上海黄金交易所数据
dotnet run sge

# 自定义ATR风险管理参数
GOLD_MACHINE_ATR_STOP_LOSS_MULTIPLIER=2.0 \
GOLD_MACHINE_ATR_TAKE_PROFIT_MULTIPLIER=3.0 \
dotnet run -- --etf 159831 --ensemble
```

## 安装

### 前置要求
- .NET 10.0 SDK 或更高版本
- 访问AKShare API（默认：http://127.0.0.1:8080/api/public）

### 构建
```bash
dotnet restore
dotnet build
```

## 配置

### 环境变量

#### 基础配置
- `GOLD_MACHINE_API_URL`: API基础URL（默认：http://127.0.0.1:8080/api/public）
- `GOLD_MACHINE_SYMBOL`: 使用的代码（默认：518880）
- `GOLD_MACHINE_START_DATE`: 开始日期，格式YYYYMMDD（默认：20000101）
- `GOLD_MACHINE_TRAIN_RATIO`: 训练数据比例0-1（默认：0.8）
- `GOLD_MACHINE_RISK_FREE_RATE`: 无风险利率，用于计算夏普比率（默认：0.02）
- `GOLD_MACHINE_DATA_PROVIDER`: 数据提供者ETF或SGE（默认：ETF）

#### 机器学习配置
- `GOLD_MACHINE_ALGORITHM`: ML算法：LinearRegression、FastTree、FastForest、OnlineGradientDescent（默认：LinearRegression）
- `GOLD_MACHINE_USE_ENSEMBLE`: 使用集成模型（组合所有算法）（默认：false）

#### FastTree参数（降低复杂度以防止过拟合）
- `GOLD_MACHINE_FASTTREE_TREES`: 树的数量（默认：30，原为100）
- `GOLD_MACHINE_FASTTREE_LEAVES`: 每棵树的叶子数（默认：10，原为20）
- `GOLD_MACHINE_FASTTREE_MIN_EXAMPLES`: 每个叶子的最小样本数（默认：50，原为10）
- `GOLD_MACHINE_FASTTREE_LEARNING_RATE`: 学习率（默认：0.1，原为0.2）
- `GOLD_MACHINE_FASTTREE_SHRINKAGE`: 收缩率（默认：0.1）

#### FastForest参数（降低复杂度以防止过拟合）
- `GOLD_MACHINE_FASTFOREST_TREES`: 树的数量（默认：30，原为100）
- `GOLD_MACHINE_FASTFOREST_LEAVES`: 每棵树的叶子数（默认：10，原为20）
- `GOLD_MACHINE_FASTFOREST_MIN_EXAMPLES`: 每个叶子的最小样本数（默认：50，原为10）
- `GOLD_MACHINE_FASTFOREST_SHRINKAGE`: 收缩率（默认：0.1）

#### ATR风险管理配置
- `GOLD_MACHINE_ATR_STOP_LOSS_MULTIPLIER`: 止损ATR倍数（默认：1.5）
- `GOLD_MACHINE_ATR_TAKE_PROFIT_MULTIPLIER`: 止盈ATR倍数（默认：2.5）
- `GOLD_MACHINE_ATR_POSITION_SIZING_ENABLED`: 启用ATR仓位管理（默认：true）
- `GOLD_MACHINE_ATR_BASE_POSITION_SIZE`: 基础仓位百分比（默认：0.2 = 20%）
- `GOLD_MACHINE_ATR_MAX_POSITION_SIZE`: 最大仓位（默认：0.3 = 30%）
- `GOLD_MACHINE_ATR_MIN_POSITION_SIZE`: 最小仓位（默认：0.05 = 5%）
- `GOLD_MACHINE_ATR_BASELINE_PERIOD`: 基准ATR计算周期（默认：30天）
- `GOLD_MACHINE_ATR_TRAILING_STOP_ENABLED`: 启用跟踪止损（默认：true）

### 命令行选项

- `--etf <symbol>`: 指定自定义ETF代码（默认：518880，GLD ETF）
- `--ensemble`: 使用集成模型（组合所有算法）
- `sge`: 使用上海黄金交易所期货数据
- 无参数：使用默认配置

## 交易策略

### 1. 基于机器学习的预测策略

核心策略使用机器学习模型预测未来价格：

- **信号生成**：预测价格 > 当前价格时买入，预测价格 < 当前价格时卖出
- **模型选择**：LinearRegression（默认）、FastTree、FastForest或集成模型
- **集成模型**：基于性能加权组合多个算法

### 2. ATR风险管理策略

使用平均真实波幅（ATR）的先进风险管理：

#### 动态止损
- **计算**：`止损 = 入场价 ± (ATR × 止损倍数)`
- **自适应**：根据市场波动性自动调整
- **跟踪止损**：价格有利时跟随移动，保护利润

#### 动态止盈
- **计算**：`止盈 = 入场价 ± (ATR × 止盈倍数)`
- **风险收益比**：默认2.5/1.5 = 1.67（每单位风险目标1.67倍收益）

#### 仓位管理
- **波动性调整**：仓位大小与当前ATR成反比
  - 低波动期（ATR < 基准）：增加仓位（最高30%）
  - 高波动期（ATR > 基准）：减少仓位（最低5%）
- **公式**：`仓位 = 基础仓位 × (基准ATR / 当前ATR)`

**示例**：
```
入场价格：9.45
当前ATR：0.15
基准ATR：0.15

止损：9.45 - (0.15 × 1.5) = 9.225
止盈：9.45 + (0.15 × 2.5) = 9.825
仓位：20%（正常波动期）
```

## 使用示例

### 基础使用
```bash
# 默认配置
dotnet run

# 自定义ETF代码
dotnet run -- --etf 159831

# 集成模型
dotnet run -- --ensemble

# 组合使用
dotnet run -- --etf 159831 --ensemble
```

### 高级配置
```bash
# 保守的ATR设置（更宽的止损）
GOLD_MACHINE_ATR_STOP_LOSS_MULTIPLIER=2.0 \
GOLD_MACHINE_ATR_TAKE_PROFIT_MULTIPLIER=3.0 \
dotnet run -- --etf 159831 --ensemble

# 禁用仓位管理（固定20%仓位）
GOLD_MACHINE_ATR_POSITION_SIZING_ENABLED=false \
dotnet run -- --etf 159831 --ensemble

# 禁用跟踪止损（仅固定止损）
GOLD_MACHINE_ATR_TRAILING_STOP_ENABLED=false \
dotnet run -- --etf 159831 --ensemble
```

## 输出与分析

### 控制台输出
应用程序提供全面的分析：

```
[INFO] 配置：API=http://127.0.0.1:8080/api/public, 代码=159831
[INFO] 数据处理成功。记录数：889
[INFO] 使用所有可用算法训练集成模型...
[INFO] 集成R²得分：0.9786
[INFO] 集成MAPE：1.00%
[INFO] ATR风险管理：止损=1.5倍ATR，止盈=2.5倍ATR
[INFO] ATR策略统计：止损触发=15，止盈触发=8，跟踪止损=3
[INFO] 仓位管理：平均=18.50%，最小=5.00%，最大=30.00%
[INFO] 回测总收益：0.01%
[INFO] 回测夏普比率：-14.39
[INFO] 回测最大回撤：0.02%
```

### 性能指标

**模型评估**：
- R²得分：决定系数
- MAE：平均绝对误差
- RMSE：均方根误差
- MAPE：平均绝对百分比误差
- sMAPE：对称MAPE
- tMAPE：截断MAPE

**策略分析**：
- 夏普比率：风险调整收益
- 胜率：盈利交易百分比
- 盈亏比：总盈利 / 总亏损
- 最大回撤：最大峰谷跌幅

**ATR统计**：
- 止损触发次数：通过止损退出的交易数
- 止盈触发次数：通过止盈退出的交易数
- 跟踪止损触发次数：通过跟踪止损退出的交易数
- 平均仓位：所有交易的平均仓位大小

### 交互式可视化

- **价格预测图表**（`price_prediction.html`）：实际价格vs预测价格，含预测区间
- **累计收益图表**（`cumulative_returns.html`）：策略表现随时间变化

## 实现细节

### 机器学习流程

1. **数据获取**：从AKShare API获取历史数据
2. **数据处理**：计算技术指标（MA、RSI、ATR等）
3. **特征工程**：为ML模型准备特征
4. **模型训练**：使用walk-forward验证训练模型
5. **集成创建**：基于性能加权组合模型
6. **预测**：生成价格预测
7. **策略执行**：应用ATR风险管理
8. **回测**：使用扩展窗口进行walk-forward回测

### ATR风险管理实现

**止损计算**：
```fsharp
let stopLoss = calculateATRStopLoss entryPrice currentATR direction multiplier
// 多头：入场价 - (ATR × 倍数)
// 空头：入场价 + (ATR × 倍数)
```

**仓位管理**：
```fsharp
let positionSize = calculateATRPositionSize currentATR baselineATR baseSize maxSize minSize
// 调整因子 = 基准ATR / 当前ATR
// 调整后仓位 = 基础仓位 × 调整因子
// 最终仓位 = 限制在(最小仓位, 最大仓位)范围内
```

**跟踪止损**：
```fsharp
let updatedPos = updateTrailingStop position currentPrice currentATR multiplier
// 新止损 = 当前价格 - (ATR × 倍数)
// 止损仅向有利方向移动
```

### Walk-Forward回测

系统实现真实的回测：

- **扩展窗口**：训练窗口随时间增长
- **样本外测试**：在未见的未来数据上测试
- **ATR集成**：回测中完整的ATR风险管理
- **交易跟踪**：记录入场/出场价格、原因和仓位大小

## 技术指标

系统计算并使用的指标：

- **移动平均**：MA3、MA9、MA20
- **动量指标**：RSI（14周期）
- **波动性指标**：ATR（14周期）、历史波动率
- **趋势指标**：MACD、EMA12、EMA26（已计算但尚未完全集成）

## 项目结构

```
gold-machine/
├── DataAcquisition.fs      # API数据获取
├── DataProcessing.fs        # 技术指标计算
├── DataProviders.fs         # 数据提供者实现
├── MachineLearning.fs       # ML模型训练和预测
├── TradingStrategy.fs       # 交易策略和ATR风险管理
├── Configuration.fs         # 配置管理
├── Types.fs                 # 类型定义
├── Visualization.fs          # 图表生成
├── Program.fs               # 主入口点
└── docs/                    # 文档
    ├── ATR_IMPLEMENTATION.md
    ├── ATR_QUANTITATIVE_ROLE.md
    ├── STRATEGY_ANALYSIS.md
    └── FIXES_SUMMARY.md
```

## 依赖项

- **Deedle**：数据处理和分析
- **MathNet.Numerics**：统计计算
- **Microsoft.ML**：机器学习框架
- **Plotly.NET**：交互式图表
- **Newtonsoft.Json**：JSON解析

## 性能改进

最近的优化：

1. **模型复杂度降低**：减少FastTree/FastForest参数以防止过拟合
   - 树数量：100 → 30
   - 叶子数：20 → 10
   - 最小样本数：10 → 50
   - 学习率：0.2 → 0.1

2. **数据泄露修复**：为集成权重计算使用独立的验证集

3. **增强指标**：添加sMAPE和截断MAPE以改进评估

4. **ATR风险管理**：动态止损、止盈和仓位管理

## 故障排除

### ATR值为零
如果ATR值全为零，请检查：
1. 数据有足够的历史记录（14周期ATR至少需要15天）
2. 最高价/最低价/收盘价有效
3. `DataProcessing.fs`中的数据对齐

### 模型性能差
- 检查是否过拟合（训练R² >> 测试R²）
- 尝试集成模型：`--ensemble`
- 通过环境变量调整模型参数

### API连接问题
- 验证API在配置的URL上运行
- 检查网络连接
- 查看API响应格式

## 贡献

欢迎贡献！请：
1. Fork仓库
2. 创建功能分支
3. 进行更改
4. 提交Pull Request

## 文档

- [ATR实现指南](docs/ATR_IMPLEMENTATION.md)
- [ATR量化作用](docs/ATR_QUANTITATIVE_ROLE.md)
- [策略分析](docs/STRATEGY_ANALYSIS.md)
- [修复总结](docs/FIXES_SUMMARY.md)

## 许可证

```
版权所有 (c) 2025 Somhairle H. Marisol

保留所有权利。

在满足以下条件的前提下，允许以源代码和二进制形式重新分发和使用（无论是否修改）：

    * 源代码的重新分发必须保留上述版权声明、
      此条件列表和以下免责声明。
    * 二进制形式的重新分发必须在文档和/或
      随软件提供的其他材料中复制上述版权声明、
      此条件列表和以下免责声明。
    * 未经事先书面许可，不得使用"Gold Machine"的名称
      或其贡献者的名称来认可或推广从本软件衍生的产品。

本软件由版权持有人和贡献者"按原样"提供，
不提供任何明示或暗示的保证，包括但不限于
适销性和特定用途适用性的暗示保证。
在任何情况下，版权所有者或贡献者均不对任何直接、
间接、偶然、特殊、惩戒性或后果性损害
（包括但不限于采购替代商品或服务；使用、数据或
利润的损失；或业务中断）承担责任，无论基于任何责任理论，
无论是合同、严格责任或侵权行为（包括疏忽或其他），
即使已被告知此类损害的可能性。
```

