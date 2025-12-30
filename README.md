# Gold Machine

A sophisticated command-line application for predicting gold ETF and futures prices using machine learning, with advanced risk management powered by ATR (Average True Range) strategies.

## Features

- **Machine Learning Prediction**: Multiple algorithms (LinearRegression, FastTree, FastForest) with ensemble support
- **ATR-Based Risk Management**: Dynamic stop-loss, take-profit, and position sizing
- **Walk-Forward Backtesting**: Realistic out-of-sample testing with expanding windows
- **Technical Indicators**: MA, RSI, ATR, MACD, Bollinger Bands, and more
- **Interactive Visualizations**: Price prediction charts and cumulative returns analysis

## Supported Data Sources

> https://akshare.akfamily.xyz/index.html

### Gold ETF Data (Default)
- API Endpoint: `fund_etf_hist_em`
- Symbol: `518880` (GLD ETF)
- Data Fields: Date, Open, High, Low, Close, Volume

### Shanghai Gold Exchange (SGE) Futures
- API Endpoint: `spot_hist_sge`
- Symbol: `Au99.99` (Gold futures)
- Data Fields: Date, Open, High, Low, Close prices

## Quick Start

```bash
# Use default ETF data (GLD ETF)
dotnet run

# Use custom ETF symbol with ensemble model
dotnet run -- --etf 159831 --ensemble

# Use Shanghai Gold Exchange data
dotnet run sge

# Custom ATR risk management
GOLD_MACHINE_ATR_STOP_LOSS_MULTIPLIER=2.0 \
GOLD_MACHINE_ATR_TAKE_PROFIT_MULTIPLIER=3.0 \
dotnet run -- --etf 159831 --ensemble
```

## Installation

### Prerequisites
- .NET 10.0 SDK or later
- Access to AKShare API (default: http://127.0.0.1:8080/api/public)

### Build
```bash
dotnet restore
dotnet build
```

## Configuration

### Environment Variables

#### Basic Configuration
- `GOLD_MACHINE_API_URL`: API base URL (default: http://127.0.0.1:8080/api/public)
- `GOLD_MACHINE_SYMBOL`: Symbol to use (default: 518880)
- `GOLD_MACHINE_START_DATE`: Start date in YYYYMMDD format (default: 20000101)
- `GOLD_MACHINE_TRAIN_RATIO`: Training data ratio 0-1 (default: 0.8)
- `GOLD_MACHINE_RISK_FREE_RATE`: Risk-free rate for Sharpe ratio (default: 0.02)
- `GOLD_MACHINE_DATA_PROVIDER`: Data provider ETF or SGE (default: ETF)

#### Machine Learning Configuration
- `GOLD_MACHINE_ALGORITHM`: ML algorithm: LinearRegression, FastTree, FastForest, OnlineGradientDescent (default: LinearRegression)
- `GOLD_MACHINE_USE_ENSEMBLE`: Use ensemble model combining all algorithms (default: false)

#### FastTree Parameters (Reduced complexity to prevent overfitting)
- `GOLD_MACHINE_FASTTREE_TREES`: Number of trees (default: 30, was 100)
- `GOLD_MACHINE_FASTTREE_LEAVES`: Number of leaves per tree (default: 10, was 20)
- `GOLD_MACHINE_FASTTREE_MIN_EXAMPLES`: Minimum examples per leaf (default: 50, was 10)
- `GOLD_MACHINE_FASTTREE_LEARNING_RATE`: Learning rate (default: 0.1, was 0.2)
- `GOLD_MACHINE_FASTTREE_SHRINKAGE`: Shrinkage (default: 0.1)

#### FastForest Parameters (Reduced complexity to prevent overfitting)
- `GOLD_MACHINE_FASTFOREST_TREES`: Number of trees (default: 30, was 100)
- `GOLD_MACHINE_FASTFOREST_LEAVES`: Number of leaves per tree (default: 10, was 20)
- `GOLD_MACHINE_FASTFOREST_MIN_EXAMPLES`: Minimum examples per leaf (default: 50, was 10)
- `GOLD_MACHINE_FASTFOREST_SHRINKAGE`: Shrinkage (default: 0.1)

#### ATR Risk Management Configuration
- `GOLD_MACHINE_ATR_STOP_LOSS_MULTIPLIER`: ATR multiplier for stop loss (default: 1.5)
- `GOLD_MACHINE_ATR_TAKE_PROFIT_MULTIPLIER`: ATR multiplier for take profit (default: 2.5)
- `GOLD_MACHINE_ATR_POSITION_SIZING_ENABLED`: Enable ATR-based position sizing (default: true)
- `GOLD_MACHINE_ATR_BASE_POSITION_SIZE`: Base position size as percentage (default: 0.2 = 20%)
- `GOLD_MACHINE_ATR_MAX_POSITION_SIZE`: Maximum position size (default: 0.3 = 30%)
- `GOLD_MACHINE_ATR_MIN_POSITION_SIZE`: Minimum position size (default: 0.05 = 5%)
- `GOLD_MACHINE_ATR_BASELINE_PERIOD`: Period for baseline ATR calculation (default: 30 days)
- `GOLD_MACHINE_ATR_TRAILING_STOP_ENABLED`: Enable trailing stop loss (default: true)

### Command Line Options

- `--etf <symbol>`: Specify custom ETF symbol (default: 518880 for GLD ETF)
- `--ensemble`: Use ensemble model (combines all algorithms)
- `sge`: Use Shanghai Gold Exchange futures data
- No arguments: Use default configuration

## Trading Strategies

### 1. ML-Based Prediction Strategy

The core strategy uses machine learning models to predict future prices:

- **Signal Generation**: Buy when predicted price > current price, Sell when predicted price < current price
- **Model Selection**: LinearRegression (default), FastTree, FastForest, or Ensemble
- **Ensemble Model**: Combines multiple algorithms with performance-based weighting

### 2. ATR Risk Management Strategy

Advanced risk management using Average True Range (ATR):

#### Dynamic Stop Loss
- **Calculation**: `StopLoss = EntryPrice ± (ATR × StopLossMultiplier)`
- **Adaptive**: Adjusts automatically based on market volatility
- **Trailing Stop**: Moves in favorable direction to protect profits

#### Dynamic Take Profit
- **Calculation**: `TakeProfit = EntryPrice ± (ATR × TakeProfitMultiplier)`
- **Risk-Reward Ratio**: Default 2.5/1.5 = 1.67 (targets 1.67x profit per unit risk)

#### Position Sizing
- **Volatility Adjustment**: Position size inversely proportional to current ATR
  - Low volatility (ATR < baseline): Increase position (up to 30%)
  - High volatility (ATR > baseline): Decrease position (down to 5%)
- **Formula**: `PositionSize = BaseSize × (BaselineATR / CurrentATR)`

**Example**:
```
Entry Price: 9.45
Current ATR: 0.15
Baseline ATR: 0.15

Stop Loss: 9.45 - (0.15 × 1.5) = 9.225
Take Profit: 9.45 + (0.15 × 2.5) = 9.825
Position Size: 20% (normal volatility)
```

## Usage Examples

### Basic Usage
```bash
# Default configuration
dotnet run

# Custom ETF symbol
dotnet run -- --etf 159831

# Ensemble model
dotnet run -- --ensemble

# Combined
dotnet run -- --etf 159831 --ensemble
```

### Advanced Configuration
```bash
# Conservative ATR settings (wider stops)
GOLD_MACHINE_ATR_STOP_LOSS_MULTIPLIER=2.0 \
GOLD_MACHINE_ATR_TAKE_PROFIT_MULTIPLIER=3.0 \
dotnet run -- --etf 159831 --ensemble

# Disable position sizing (fixed 20% position)
GOLD_MACHINE_ATR_POSITION_SIZING_ENABLED=false \
dotnet run -- --etf 159831 --ensemble

# Disable trailing stop (fixed stop loss only)
GOLD_MACHINE_ATR_TRAILING_STOP_ENABLED=false \
dotnet run -- --etf 159831 --ensemble
```

## Output & Analytics

### Console Output
The application provides comprehensive analysis:

```
[INFO] Configuration: API=http://127.0.0.1:8080/api/public, Symbol=159831
[INFO] Data processed successfully. Records: 889
[INFO] Training ensemble model with all available algorithms...
[INFO] Ensemble R² Score: 0.9786
[INFO] Ensemble MAPE: 1.00%
[INFO] ATR Risk Management: StopLoss=1.5x ATR, TakeProfit=2.5x ATR
[INFO] ATR Strategy Statistics: StopLoss=15, TakeProfit=8, TrailingStop=3
[INFO] Position Sizing: Avg=18.50%, Min=5.00%, Max=30.00%
[INFO] Backtest Total Return: 0.01%
[INFO] Backtest Sharpe Ratio: -14.39
[INFO] Backtest Max Drawdown: 0.02%
```

### Performance Metrics

**Model Evaluation**:
- R² Score: Coefficient of determination
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- sMAPE: Symmetric MAPE
- tMAPE: Truncated MAPE

**Strategy Analysis**:
- Sharpe Ratio: Risk-adjusted return
- Win Rate: Percentage of profitable trades
- Profit Factor: Gross profit / Gross loss
- Maximum Drawdown: Largest peak-to-trough decline

**ATR Statistics**:
- Stop Loss Triggers: Number of trades exited via stop loss
- Take Profit Triggers: Number of trades exited via take profit
- Trailing Stop Triggers: Number of trades exited via trailing stop
- Average Position Size: Mean position size across all trades

### Interactive Visualizations

- **Price Prediction Chart** (`price_prediction.html`): Actual vs predicted prices with prediction intervals
- **Cumulative Returns Chart** (`cumulative_returns.html`): Strategy performance over time

## Implementation Details

### Machine Learning Pipeline

1. **Data Acquisition**: Fetch historical data from AKShare API
2. **Data Processing**: Calculate technical indicators (MA, RSI, ATR, etc.)
3. **Feature Engineering**: Prepare features for ML models
4. **Model Training**: Train models with walk-forward validation
5. **Ensemble Creation**: Combine models with performance-based weights
6. **Prediction**: Generate price predictions
7. **Strategy Execution**: Apply ATR risk management
8. **Backtesting**: Walk-forward backtesting with expanding windows

### ATR Risk Management Implementation

**Stop Loss Calculation**:
```fsharp
let stopLoss = calculateATRStopLoss entryPrice currentATR direction multiplier
// For long: EntryPrice - (ATR × multiplier)
// For short: EntryPrice + (ATR × multiplier)
```

**Position Sizing**:
```fsharp
let positionSize = calculateATRPositionSize currentATR baselineATR baseSize maxSize minSize
// Adjustment factor = BaselineATR / CurrentATR
// Adjusted size = BaseSize × AdjustmentFactor
// Final size = clamp(AdjustedSize, MinSize, MaxSize)
```

**Trailing Stop**:
```fsharp
let updatedPos = updateTrailingStop position currentPrice currentATR multiplier
// New stop = CurrentPrice - (ATR × multiplier)
// Stop only moves in favorable direction
```

### Walk-Forward Backtesting

The system implements realistic backtesting:

- **Expanding Windows**: Training window grows over time
- **Out-of-Sample Testing**: Test on unseen future data
- **ATR Integration**: Full ATR risk management in backtest
- **Trade Tracking**: Records entry/exit prices, reasons, and position sizes

## Technical Indicators

The system calculates and uses:

- **Moving Averages**: MA3, MA9, MA20
- **Momentum**: RSI (14-period)
- **Volatility**: ATR (14-period), Historical Volatility
- **Trend**: MACD, EMA12, EMA26 (calculated but not yet fully integrated)

## Project Structure

```
gold-machine/
├── DataAcquisition.fs      # API data fetching
├── DataProcessing.fs        # Technical indicators calculation
├── DataProviders.fs         # Data provider implementations
├── MachineLearning.fs       # ML model training and prediction
├── TradingStrategy.fs       # Trading strategies and ATR risk management
├── Configuration.fs         # Configuration management
├── Types.fs                 # Type definitions
├── Visualization.fs          # Chart generation
├── Program.fs               # Main entry point
└── docs/                    # Documentation
    ├── ATR_IMPLEMENTATION.md
    ├── ATR_QUANTITATIVE_ROLE.md
    ├── STRATEGY_ANALYSIS.md
    └── FIXES_SUMMARY.md
```

## Dependencies

- **Deedle**: Data manipulation and analysis
- **MathNet.Numerics**: Statistical computations
- **Microsoft.ML**: Machine learning framework
- **Plotly.NET**: Interactive charting
- **Newtonsoft.Json**: JSON parsing

## Performance Improvements

Recent optimizations:

1. **Model Complexity Reduction**: Reduced FastTree/FastForest parameters to prevent overfitting
   - Trees: 100 → 30
   - Leaves: 20 → 10
   - Min Examples: 10 → 50
   - Learning Rate: 0.2 → 0.1

2. **Data Leakage Fix**: Separate validation set for ensemble weighting

3. **Enhanced Metrics**: Added sMAPE and truncated MAPE for better evaluation

4. **ATR Risk Management**: Dynamic stop-loss, take-profit, and position sizing

## Troubleshooting

### ATR Values Are Zero
If ATR values are all zero, check:
1. Data has sufficient history (need at least 15 days for 14-period ATR)
2. High/Low/Close prices are valid
3. Data alignment in `DataProcessing.fs`

### Poor Model Performance
- Check for overfitting (training R² >> test R²)
- Try ensemble model: `--ensemble`
- Adjust model parameters via environment variables

### API Connection Issues
- Verify API is running at configured URL
- Check network connectivity
- Review API response format

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Documentation

- [ATR Implementation Guide](docs/ATR_IMPLEMENTATION.md)
- [ATR Quantitative Role](docs/ATR_QUANTITATIVE_ROLE.md)
- [Strategy Analysis](docs/STRATEGY_ANALYSIS.md)
- [Fixes Summary](docs/FIXES_SUMMARY.md)

## License

```
Copyright (c) 2025 Somhairle H. Marisol

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of "Gold Machine" nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
