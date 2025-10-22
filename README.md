# Gold Machine

A command-line application for predicting gold ETF and futures prices using machine learning.

## Supported Data Sources
> https://akshare.akfamily.xyz/index.html

### Gold ETF Data (Default)
- API Endpoint: `fund_etf_hist_em`
- Symbol: `518880` (GLD ETF)
- Data Fields: Date, Close price

### Shanghai Gold Exchange (SGE) Futures
- API Endpoint: `spot_hist_sge`
- Symbol: `Au99.99` (Gold futures)
- Data Fields: Date, Open, High, Low, Close prices

## Dependencies

- Deedle: Data manipulation and analysis
- MathNet.Numerics: Statistical computations
- Microsoft.ML: Machine learning framework
- Plotly.NET: Interactive charting
- Newtonsoft.Json: JSON parsing

## Usage

```bash
# Use default ETF data (GLD ETF)
dotnet run

# Use custom ETF symbol
dotnet run --etf 159549

# Use Shanghai Gold Exchange data
dotnet run sge

# Set configuration via environment variables
GOLD_MACHINE_SYMBOL=159549 GOLD_MACHINE_ALGORITHM=FastTree dotnet run
```

### Configuration Options

The system supports configuration via environment variables:

- `GOLD_MACHINE_API_URL`: API base URL (default: http://127.0.0.1:8080/api/public)
- `GOLD_MACHINE_SYMBOL`: Symbol to use (default: 518880)
- `GOLD_MACHINE_START_DATE`: Start date in YYYYMMDD format (default: 20000101)
- `GOLD_MACHINE_TRAIN_RATIO`: Training data ratio 0-1 (default: 0.8)
- `GOLD_MACHINE_RISK_FREE_RATE`: Risk-free rate for Sharpe ratio (default: 0.02)
- `GOLD_MACHINE_DATA_PROVIDER`: Data provider ETF or SGE (default: ETF)
- `GOLD_MACHINE_ALGORITHM`: ML algorithm: LinearRegression, FastTree, FastForest, OnlineGradientDescent (default: LinearRegression)

### Command Line Options

- `--etf <symbol>`: Specify custom ETF symbol (default: 518880 for GLD ETF)
- `sge`: Use Shanghai Gold Exchange futures data
- No arguments: Use default ETF data

### Examples

```bash
# Default usage - GLD ETF
dotnet run

# Custom ETF symbol
dotnet run --etf 159549

# Shanghai Gold Exchange futures
dotnet run sge

# Use FastTree algorithm
GOLD_MACHINE_ALGORITHM=FastTree dotnet run

# Custom configuration
GOLD_MACHINE_SYMBOL=159549 GOLD_MACHINE_TRAIN_RATIO=0.9 dotnet run
```

## Configuration

The system uses environment variables for configuration with sensible defaults. The configuration is validated at startup to ensure data integrity.

### Default Configuration
- API Base URL: `http://127.0.0.1:8080/api/public`
- Symbol: `518880` (GLD ETF)
- Start Date: `20000101` (January 1, 2000)
- Training Ratio: `0.8` (80% training, 20% testing)
- Risk-Free Rate: `0.02` (2% annual risk-free rate)
- Data Provider: `ETF`
- ML Algorithm: `LinearRegression`

## Adding New Data Sources

To add a new data source:

1. Define Data Types in `Types.fs`:
   ```fsharp
   type RawNewDataSource =
     { [<JsonProperty("field")>]
       Field : string }

   type NewDataSourceResponse = RawNewDataSource[]
   ```

2. Update RawDataSource Union:
   ```fsharp
   type RawDataSource =
     | ETF of RawGoldETFData[]
     | SGE of RawGoldSGEData[]
     | NewSource of RawNewDataSource[]  // Add new case
   ```

3. Add DataProviderType:
   ```fsharp
   type DataProviderType =
     | ETFProvider
     | SGEProvider
     | NewProvider  // Add new provider type
   ```

4. Implement Data Provider in `DataProviders.fs`:
   ```fsharp
   type NewDataProvider() =
       interface IDataProvider with
           member this.Name = "New Data Provider"
           member this.ProviderType = NewProvider
           member this.FetchRawData config = // Implementation
   ```

5. Update Data Processing in `DataProcessing.fs`:
   ```fsharp
   let convertRawDataToRecords rawData =
     match rawData with
     | NewSource data -> // Handle new data format
   ```

6. Add Configuration in `Configuration.fs`:
   ```fsharp
   let getNewConfig () = // New configuration function
   ```

## Output & Analytics

### Console Output
The application provides comprehensive analysis output:

```
Gold Price Prediction System v2.0
===================================
Usage: dotnet run [options]
[... configuration info ...]

[INFO] Configuration: API=http://127.0.0.1:8080/api/public, Symbol=518880, StartDate=20000101
[INFO] Acquiring gold price data from Gold ETF Provider...
[INFO] Data processed successfully. Records: 2966
[INFO] Training LinearRegression model...
[INFO] Model R² Score: 0.9982
[INFO] Model MAE: 0.0333
[INFO] Model RMSE: 0.0521
[INFO] Model MAPE: 0.54%
[INFO] Strategy Sharpe Ratio: -2.1719
[INFO] Performing walk-forward backtesting...
[INFO] Backtest Total Return: -216.04%
[INFO] Backtest Annualized Return: -20.04%
[INFO] Backtest Sharpe Ratio: -3.0511
[INFO] Backtest Max Drawdown: 316.84%
[INFO] Trading recommendation: BUY GLD - Predicted price higher than current price
```

### Performance Metrics
- Model Evaluation: R², MAE, RMSE, MAPE scores
- Strategy Analysis: Sharpe ratio, win rate, profit factor
- Backtesting Results: Total return, annualized return, maximum drawdown
- Trading Signals: Buy/Hold/Sell recommendations based on predictions

### Data Quality Assurance
- Outlier detection using IQR method
- Chronological data validation
- Missing data imputation
- Anomaly removal using statistical filters

### Interactive Visualizations
- Price Prediction Chart (`gold_price_prediction.html`): Actual vs predicted prices with prediction intervals
- Cumulative Returns Chart (`cumulative_returns.html`): Strategy performance over time

### Backtesting Framework
The system implements walk-forward backtesting with:
- Expanding training windows
- Rolling out-of-sample testing
- Realistic position sizing and trade execution
- Comprehensive risk metrics calculation

## Machine Learning Pipeline

### 1. Configuration & Validation
- Load configuration from environment variables with validation
- Support multiple data providers (ETF/SGE) and ML algorithms
- Ensure data integrity and parameter consistency

### 2. Data Acquisition
- Fetch historical price data from external APIs
- Support configurable symbols and date ranges
- Handle API errors gracefully with retry logic

### 3. Data Quality Assurance
- Outlier detection using statistical methods (IQR)
- Chronological ordering validation
- Anomaly removal and data cleaning
- Missing value imputation

### 4. Feature Engineering
- Calculate technical indicators (MA3, MA9 moving averages)
- Ensure proper data alignment and validation
- Handle edge cases in moving average calculations

### 5. Model Training & Selection
- Support multiple ML algorithms: LinearRegression, FastTree, FastForest, OnlineGradientDescent
- Cross-validation for model evaluation
- Automatic algorithm selection based on performance

### 6. Model Evaluation
- Comprehensive metrics: R², MAE, RMSE, MAPE
- Model health assessment and risk classification
- Prediction interval estimation

### 7. Trading Strategy
- Simple momentum-based signals (Buy/Hold/Sell)
- Sharpe ratio calculation and risk assessment
- Trading recommendation generation

### 8. Walk-Forward Backtesting
- Realistic out-of-sample testing with expanding windows
- Comprehensive risk metrics (total return, max drawdown, win rate)
- Position sizing and trade execution simulation

### 9. Visualization & Reporting
- Interactive price prediction charts
- Cumulative returns analysis
- HTML output for web viewing

### 10. Error Handling & Logging
- Comprehensive error handling with Result types
- Detailed logging throughout the pipeline
- Graceful failure recovery

## [LICENSE](./LICENSE)

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
