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
# Use default ETF data
dotnet run

# Use Shanghai Gold Exchange data
dotnet run sge
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

## Output

The system generates:
- Console logs with detailed progress information
- Performance metrics (R-squared, Sharpe ratio)
- Trading recommendations for the latest data
- Interactive HTML charts:
  - `gold_price_prediction.html`: Actual vs predicted prices
  - `cumulative_returns.html`: Strategy performance

## Machine Learning Pipeline

1. Data Acquisition: Fetch historical price data from selected provider
2. Data Processing: Convert raw data to standardized format and calculate technical indicators
3. Data Splitting: 80/20 train/test split
4. Model Training: Linear regression using SDCA algorithm
5. Prediction: Generate price forecasts
6. Strategy Evaluation: Calculate returns and risk metrics
7. Visualization: Create performance charts

## Trading Strategy

The system implements a simple momentum strategy:
- Buy Signal: When predicted price > actual price
- Hold Signal: When predicted price ≤ actual price

Strategy performance is evaluated using:
- Cumulative returns
- Sharpe ratio (risk-adjusted performance)
- Win rate analysis


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
