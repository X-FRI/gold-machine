namespace GoldMachine

open System
open Microsoft.ML.Data
open Newtonsoft.Json

/// <summary>
/// Represents the input data for price prediction models.
/// Contains carefully selected technical indicators as features for machine learning.
/// </summary>
[<CLIMutable>]
type PredictionInput =
  { [<LoadColumn(0)>]
    MA3 : float32
    [<LoadColumn(1)>]
    MA9 : float32
    [<LoadColumn(2)>]
    MA20 : float32
    [<LoadColumn(3)>]
    RSI : float32
    [<LoadColumn(4)>]
    ATR : float32
    [<LoadColumn(5)>]
    Volatility : float32
    [<LoadColumn(6)>]
    ChangePercent : float32 }

/// <summary>
/// Represents the output from a price prediction model.
/// Contains the predicted price score.
/// </summary>
[<CLIMutable>]
type PredictionOutput = { Score : float32 }

/// <summary>
/// Represents training data for machine learning models.
/// Includes carefully selected technical indicators and the target label (actual price).
/// </summary>
[<CLIMutable>]
type TrainingData =
  { [<LoadColumn(0)>]
    MA3 : float32
    [<LoadColumn(1)>]
    MA9 : float32
    [<LoadColumn(2)>]
    MA20 : float32
    [<LoadColumn(3)>]
    RSI : float32
    [<LoadColumn(4)>]
    ATR : float32
    [<LoadColumn(5)>]
    Volatility : float32
    [<LoadColumn(6)>]
    ChangePercent : float32
    [<LoadColumn(7)>]
    Label : float32 }

/// <summary>
/// Parameters for FastTree regression algorithm.
/// </summary>
type FastTreeParameters =
  { NumberOfTrees : int
    NumberOfLeaves : int
    MinimumExampleCountPerLeaf : int
    LearningRate : float32
    Shrinkage : float32 }

/// <summary>
/// Parameters for FastForest regression algorithm.
/// </summary>
type FastForestParameters =
  { NumberOfTrees : int
    NumberOfLeaves : int
    MinimumExampleCountPerLeaf : int
    Shrinkage : float32 }

/// <summary>
/// Supported machine learning algorithms for price prediction.
/// </summary>
type MLAlgorithm =
  | LinearRegression
  | FastTreeRegression of FastTreeParameters
  | FastForestRegression of FastForestParameters
  | OnlineGradientDescentRegression

/// <summary>
/// Represents a trained machine learning model for gold price prediction.
/// Encapsulates the ML context, trained transformer, and data schema.
/// </summary>
type GoldPredictionModel =
  { MLContext : Microsoft.ML.MLContext
    Model : Microsoft.ML.ITransformer
    Schema : Microsoft.ML.DataViewSchema
    Algorithm : MLAlgorithm }

/// <summary>
/// Represents an ensemble model that combines multiple individual models.
/// </summary>
type EnsembleModel =
  { Models : GoldPredictionModel list
    Weights : float list
    MLContext : Microsoft.ML.MLContext }

/// <summary>
/// Represents evaluation metrics for model performance.
/// Contains R-squared value and other statistical measures.
/// </summary>
type ModelEvaluation =
  { RSquared : float32
    SharpeRatio : float
    MAE : float32
    RMSE : float32
    MAPE : float }

/// <summary>
/// Represents the result of ensemble model evaluation.
/// </summary>
type EnsembleEvaluation =
  { IndividualEvaluations : (MLAlgorithm * ModelEvaluation) list
    EnsembleRSquared : float32
    EnsembleMAE : float32
    EnsembleRMSE : float32
    EnsembleMAPE : float
    SharpeRatio : float }

/// <summary>
/// Represents a single record of gold price data with comprehensive technical indicators.
/// Contains OHLCV data and calculated technical indicators for advanced analysis.
/// </summary>
type GoldDataRecord =
  { Date : DateTime
    Open : float
    High : float
    Low : float
    Close : float
    Volume : int64
    Amount : float
    // Moving Averages
    MA3 : float32
    MA5 : float32
    MA9 : float32
    MA20 : float32
    EMA12 : float32
    EMA26 : float32
    // Momentum Indicators
    RSI : float32
    MACD : float32
    MACDSignal : float32
    MACDHistogram : float32
    // Volatility Indicators
    ATR : float32
    BollingerUpper : float32
    BollingerMiddle : float32
    BollingerLower : float32
    Volatility : float32
    // Price Change Indicators
    ChangePercent : float32
    ChangeAmount : float32
    // Volume Indicators
    OBV : float32
    VWAP : float32 }

/// <summary>
/// Represents raw data retrieved from the gold ETF API.
/// Contains comprehensive OHLCV and derived metrics.
/// </summary>
type RawGoldETFData =
  { [<JsonProperty("日期")>]
    Date : string
    [<JsonProperty("开盘")>]
    Open : float
    [<JsonProperty("最高")>]
    High : float
    [<JsonProperty("最低")>]
    Low : float
    [<JsonProperty("收盘")>]
    Close : float
    [<JsonProperty("成交量")>]
    Volume : int64
    [<JsonProperty("成交额")>]
    Amount : float
    [<JsonProperty("振幅")>]
    Amplitude : float
    [<JsonProperty("涨跌幅")>]
    ChangePercent : float
    [<JsonProperty("涨跌额")>]
    ChangeAmount : float
    [<JsonProperty("换手率")>]
    TurnoverRate : float }

/// <summary>
/// Represents raw data retrieved from the Shanghai Gold Exchange API.
/// Contains OHLC (Open, High, Low, Close) price information.
/// </summary>
type RawGoldSGEData =
  { [<JsonProperty("date")>]
    Date : DateTime
    [<JsonProperty("open")>]
    Open : float
    [<JsonProperty("high")>]
    High : float
    [<JsonProperty("low")>]
    Low : float
    [<JsonProperty("close")>]
    Close : float }

/// <summary>
/// Represents the response structure from the gold ETF API.
/// The API returns a direct array of gold data records.
/// </summary>
type ETFResponse = RawGoldETFData[]

/// <summary>
/// Represents the response structure from the Shanghai Gold Exchange API.
/// The API returns a direct array of OHLC data records.
/// </summary>
type SGEResponse = RawGoldSGEData[]

/// <summary>
/// Union type representing different types of raw data sources.
/// </summary>
type RawDataSource =
  | ETF of RawGoldETFData[]
  | SGE of RawGoldSGEData[]

/// <summary>
/// Enumeration of supported data providers.
/// </summary>
type DataProviderType =
  | ETFProvider
  | SGEProvider

/// <summary>
/// Represents different types of errors that can occur in the system.
/// </summary>
type GoldMachineError =
  | InvalidDateRange of string
  | DataAcquisitionFailed of string
  | ModelTrainingFailed of string
  | FileOperationFailed of string
  | ConfigurationError of string

/// <summary>
/// ATR-based risk management configuration.
/// </summary>
type ATRRiskConfig =
  { /// <summary>
    /// Multiplier for ATR-based stop loss (default: 1.5-2.0).
    /// Stop loss distance = ATR × StopLossMultiplier
    /// </summary>
    StopLossMultiplier : float
    /// <summary>
    /// Multiplier for ATR-based take profit (default: 2.0-3.0).
    /// Take profit distance = ATR × TakeProfitMultiplier
    /// </summary>
    TakeProfitMultiplier : float
    /// <summary>
    /// Whether to enable ATR-based position sizing (default: true).
    /// </summary>
    PositionSizingEnabled : bool
    /// <summary>
    /// Base position size as percentage of capital (default: 0.2 = 20%).
    /// </summary>
    BasePositionSize : float
    /// <summary>
    /// Maximum position size as percentage of capital (default: 0.3 = 30%).
    /// </summary>
    MaxPositionSize : float
    /// <summary>
    /// Minimum position size as percentage of capital (default: 0.05 = 5%).
    /// </summary>
    MinPositionSize : float
    /// <summary>
    /// Period for calculating ATR baseline (default: 30 days).
    /// </summary>
    ATRBaselinePeriod : int
    /// <summary>
    /// Whether to use trailing stop loss (default: true).
    /// </summary>
    TrailingStopEnabled : bool }

/// <summary>
/// Configuration settings for the gold price prediction system.
/// Includes API endpoints, data parameters, and model settings.
/// </summary>
type GoldMachineConfig =
  { ApiBaseUrl : string
    Symbol : string
    StartDate : string
    TrainRatio : float
    RiskFreeRate : float
    DataProvider : DataProviderType
    MLAlgorithm : MLAlgorithm
    UseEnsemble : bool
    FastTreeParams : FastTreeParameters
    FastForestParams : FastForestParameters
    ATRRiskConfig : ATRRiskConfig }

/// <summary>
/// Abstract interface for data providers.
/// Defines the contract for different data sources to implement.
/// </summary>
type IDataProvider =
  /// <summary>
  /// Gets the name of the data provider.
  /// </summary>
  abstract member Name : string

  /// <summary>
  /// Gets the type of the data provider.
  /// </summary>
  abstract member ProviderType : DataProviderType

  /// <summary>
  /// Fetches raw data from the data source.
  /// </summary>
  /// <param name="config">Configuration containing data source parameters.</param>
  /// <returns>Result containing raw data or an error.</returns>
  abstract member FetchRawData :
    GoldMachineConfig -> Async<Result<RawDataSource, GoldMachineError>>

/// <summary>
/// Represents trading signals generated by the strategy.
/// Values: 1.0 for buy signal, 0.0 for hold.
/// </summary>
type TradingSignal = float

/// <summary>
/// Represents different health statuses for model evaluation.
/// </summary>
type ModelHealthStatus =
  | Normal
  | Degrading
  | OutlierDetected
  | Critical

/// <summary>
/// Represents model health assessment results.
/// </summary>
type ModelHealthReport =
  { Status : ModelHealthStatus
    Message : string
    Recommendations : string list
    RiskLevel : float }
