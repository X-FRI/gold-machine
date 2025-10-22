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
dotnet run --etf 159941

# Use Shanghai Gold Exchange data
dotnet run sge
```

### Command Line Options

- `--etf <symbol>`: Specify custom ETF symbol (default: 518880 for GLD ETF)
- `sge`: Use Shanghai Gold Exchange futures data
- No arguments: Use default ETF data

### Examples

```bash
# Default usage - GLD ETF
dotnet run

# Custom ETF - e.g., another gold ETF
dotnet run --etf 159941

# Shanghai Gold Exchange futures
dotnet run sge

# Help information
dotnet run --help
```

## Configuration

The system supports different configurations for each data provider:

### ETF Configuration (Default)
- API Base URL: `http://127.0.0.1:8080/api/public`
- Symbol: `518880`
- Start Date: `20000101`
- Training Ratio: 80%
- Risk-Free Rate: 0%

### SGE Configuration
- API Base URL: `http://127.0.0.1:8080/api/public`
- Symbol: `Au99.99`
- Start Date: `20000101`
- Training Ratio: 80%
- Risk-Free Rate: 0%

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

### Console Analytics
- Model Diagnostics: Real-time health assessment with risk level classification
- Performance Metrics: R², MAE, RMSE, MAPE with trend analysis
- Strategy Analytics: Sharpe ratio, position sizing recommendations, market regime classification
- Risk Management: Kelly Criterion sizing, stop-loss levels, cost analysis
- Market Intelligence: Volatility assessment, prediction intervals, data quality monitoring
- Health Recommendations: Automated suggestions for model retraining and parameter adjustments

### Performance Reporting
- Traditional Metrics: Cumulative returns, maximum drawdown, win rate analysis
- Advanced Metrics: Risk-adjusted performance using MAE/RMSE uncertainty measures
- Cost-Adjusted Results: Net performance after all trading costs and slippage
- Model Correlation: Strategy performance linked to model health indicators

### Trading Intelligence
- Signal Confidence: Weighted signals with uncertainty quantification
- Position Recommendations: Dynamic sizing based on multiple criteria
- Market Adaptation: Real-time parameter adjustments based on regime changes
- Risk Alerts: Automated notifications for deteriorating conditions

### Interactive Visualizations
- Enhanced Price Charts: `gold_price_prediction.html` with prediction intervals and uncertainty bands
- Advanced Performance Charts: `cumulative_returns.html` with risk metrics and drawdown analysis
- Strategy Analytics: Visual representation of signal confidence and position sizing
- Model Health Dashboard: Graphical display of model diagnostics and trend analysis

## Machine Learning Pipeline

1. Data Acquisition: Fetch historical price data from configured providers (ETF/SGE)
2. Data Processing: Convert raw data to standardized format with technical indicators (MA3, MA9)
3. Data Splitting: 80/20 train/test split with temporal ordering preservation
4. Model Training: Linear regression using SDCA algorithm with feature engineering
5. Model Evaluation: Comprehensive metrics calculation (R², MAE, RMSE, MAPE)
6. Model Health Assessment: Real-time diagnostics with degradation detection and risk classification
7. Trend Monitoring: Performance trend analysis with automated model health recommendations
8. Advanced Signal Generation: Confidence-weighted signals with market regime classification
9. Adaptive Strategy Execution: Dynamic position sizing based on MAE thresholds and market conditions
10. Ensemble Learning: Multi-model weighting with uncertainty quantification and prediction intervals
11. Risk Management: Kelly Criterion optimization with dynamic stop-loss and cost analysis
12. Market Adaptation: Real-time parameter adjustment based on volatility regimes
13. Data Quality Monitoring: Continuous validation with anomaly detection and trading gates
14. Prediction Generation: Price forecasts with statistical uncertainty bounds
15. Strategy Evaluation: Multi-dimensional performance assessment including trading costs
16. Visualization: Interactive charts with advanced analytics and risk metrics

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
