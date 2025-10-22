namespace GoldMachine

open System
open System.IO

/// <summary>
/// Configuration management for the Gold Price Prediction System.
/// Provides environment-based settings and validation for system parameters.
/// </summary>
module Configuration =

  /// <summary>
  /// Gets a value from environment variables with a fallback default.
  /// </summary>
  /// <param name="key">Environment variable key.</param>
  /// <param name="defaultValue">Default value if environment variable is not set.</param>
  /// <returns>The environment variable value or default.</returns>
  let getEnvVar (key : string) (defaultValue : string) =
    let value = Environment.GetEnvironmentVariable key
    if String.IsNullOrWhiteSpace value then defaultValue else value

  /// <summary>
  /// Gets a numeric value from environment variables with a fallback default.
  /// </summary>
  /// <param name="key">Environment variable key.</param>
  /// <param name="defaultValue">Default numeric value.</param>
  /// <returns>The parsed numeric value or default.</returns>
  let getEnvVarFloat (key : string) (defaultValue : float) =
    let value = getEnvVar key ""

    match Double.TryParse value with
    | true, parsed -> parsed
    | false, _ -> defaultValue

  /// <summary>
  /// Loads configuration from environment variables and config file.
  /// </summary>
  /// <returns>GoldMachineConfig loaded from various sources.</returns>
  let loadConfig () =
    // Load from environment variables with defaults
    let apiBaseUrl =
      getEnvVar "GOLD_MACHINE_API_URL" "http://127.0.0.1:8080/api/public"

    let symbol = getEnvVar "GOLD_MACHINE_SYMBOL" "518880"
    let startDate = getEnvVar "GOLD_MACHINE_START_DATE" "20000101"
    let trainRatio = getEnvVarFloat "GOLD_MACHINE_TRAIN_RATIO" 0.8
    let riskFreeRate = getEnvVarFloat "GOLD_MACHINE_RISK_FREE_RATE" 0.02

    let dataProviderType =
      match getEnvVar "GOLD_MACHINE_DATA_PROVIDER" "ETF" with
      | "SGE" -> SGEProvider
      | _ -> ETFProvider

    let algorithm =
      match getEnvVar "GOLD_MACHINE_ALGORITHM" "LinearRegression" with
      | "FastTree" ->
        let fastTreeParameters =
          { NumberOfTrees = int (getEnvVar "GOLD_MACHINE_FASTTREE_TREES" "100")
            NumberOfLeaves = int (getEnvVar "GOLD_MACHINE_FASTTREE_LEAVES" "20")
            MinimumExampleCountPerLeaf =
              int (getEnvVar "GOLD_MACHINE_FASTTREE_MIN_EXAMPLES" "10")
            LearningRate =
              float32 (getEnvVarFloat "GOLD_MACHINE_FASTTREE_LEARNING_RATE" 0.2)
            Shrinkage =
              float32 (getEnvVarFloat "GOLD_MACHINE_FASTTREE_SHRINKAGE" 0.1) }

        FastTreeRegression fastTreeParameters
      | "FastForest" ->
        let fastForestParameters =
          { NumberOfTrees =
              int (getEnvVar "GOLD_MACHINE_FASTFOREST_TREES" "100")
            NumberOfLeaves =
              int (getEnvVar "GOLD_MACHINE_FASTFOREST_LEAVES" "20")
            MinimumExampleCountPerLeaf =
              int (getEnvVar "GOLD_MACHINE_FASTFOREST_MIN_EXAMPLES" "10")
            Shrinkage =
              float32 (getEnvVarFloat "GOLD_MACHINE_FASTFOREST_SHRINKAGE" 0.1) }

        FastForestRegression fastForestParameters
      | "OnlineGradientDescent" -> OnlineGradientDescentRegression
      | _ -> LinearRegression

    let useEnsemble =
      match getEnvVar "GOLD_MACHINE_USE_ENSEMBLE" "false" with
      | "true" -> true
      | _ -> false

    // FastTree parameters
    let fastTreeParameters =
      { NumberOfTrees = int (getEnvVar "GOLD_MACHINE_FASTTREE_TREES" "100")
        NumberOfLeaves = int (getEnvVar "GOLD_MACHINE_FASTTREE_LEAVES" "20")
        MinimumExampleCountPerLeaf =
          int (getEnvVar "GOLD_MACHINE_FASTTREE_MIN_EXAMPLES" "10")
        LearningRate =
          float32 (getEnvVarFloat "GOLD_MACHINE_FASTTREE_LEARNING_RATE" 0.2)
        Shrinkage =
          float32 (getEnvVarFloat "GOLD_MACHINE_FASTTREE_SHRINKAGE" 0.1) }

    // FastForest parameters
    let fastForestParameters =
      { NumberOfTrees = int (getEnvVar "GOLD_MACHINE_FASTFOREST_TREES" "100")
        NumberOfLeaves = int (getEnvVar "GOLD_MACHINE_FASTFOREST_LEAVES" "20")
        MinimumExampleCountPerLeaf =
          int (getEnvVar "GOLD_MACHINE_FASTFOREST_MIN_EXAMPLES" "10")
        Shrinkage =
          float32 (getEnvVarFloat "GOLD_MACHINE_FASTFOREST_SHRINKAGE" 0.1) }

    { ApiBaseUrl = apiBaseUrl
      Symbol = symbol
      StartDate = startDate
      TrainRatio = trainRatio
      RiskFreeRate = riskFreeRate
      DataProvider = dataProviderType
      MLAlgorithm = algorithm
      UseEnsemble = useEnsemble
      FastTreeParams = fastTreeParameters
      FastForestParams = fastForestParameters }

  /// <summary>
  /// Gets the default configuration for the gold price prediction system.
  /// </summary>
  /// <returns>A GoldMachineConfig with default values.</returns>
  let getDefaultConfig () = loadConfig ()

  /// <summary>
  /// Gets the configuration for Shanghai Gold Exchange data.
  /// </summary>
  /// <returns>A GoldMachineConfig configured for SGE data.</returns>
  let getSGEConfig () =
    // Override environment config for SGE
    let baseConfig = loadConfig ()

    { baseConfig with
        Symbol = "Au99.99"
        DataProvider = SGEProvider }

  /// <summary>
  /// Gets the configuration for a custom ETF symbol.
  /// </summary>
  /// <param name="etfSymbol">The ETF symbol to use (e.g., "518880" for GLD ETF).</param>
  /// <returns>A GoldMachineConfig configured for the specified ETF.</returns>
  let getETFConfig (etfSymbol : string) =
    if String.IsNullOrWhiteSpace etfSymbol then
      getDefaultConfig () // Fall back to default if symbol is empty
    else
      let baseConfig = loadConfig ()

      { baseConfig with
          Symbol = etfSymbol
          DataProvider = ETFProvider }

  /// <summary>
  /// Validates the configuration parameters.
  /// </summary>
  /// <param name="config">The configuration to validate.</param>
  /// <returns>Result indicating success or validation error.</returns>
  let validateConfig (config : GoldMachineConfig) =
    if String.IsNullOrWhiteSpace (config.ApiBaseUrl) then
      Error (ConfigurationError "API base URL cannot be empty")
    elif
      not (
        config.ApiBaseUrl.StartsWith "http://"
        || config.ApiBaseUrl.StartsWith "https://"
      )
    then
      Error (
        ConfigurationError "API base URL must start with http:// or https://"
      )
    elif String.IsNullOrWhiteSpace (config.Symbol) then
      Error (ConfigurationError "Symbol cannot be empty")
    elif String.IsNullOrWhiteSpace (config.StartDate) then
      Error (ConfigurationError "Start date cannot be empty")
    elif
      config.StartDate.Length <> 8
      || not (
        System.Text.RegularExpressions.Regex.IsMatch (
          config.StartDate,
          @"^\d{8}$"
        )
      )
    then
      Error (ConfigurationError "Start date must be in YYYYMMDD format")
    elif config.TrainRatio <= 0.0 || config.TrainRatio >= 1.0 then
      Error (
        ConfigurationError "Train ratio must be between 0 and 1 (exclusive)"
      )
    elif config.RiskFreeRate < 0.0 || config.RiskFreeRate > 1.0 then
      Error (ConfigurationError "Risk-free rate must be between 0 and 1")
    else
      Ok config

  /// <summary>
  /// Constructs the full API URL for fetching gold ETF data.
  /// </summary>
  /// <param name="config">The configuration containing API parameters.</param>
  /// <returns>The complete API endpoint URL.</returns>
  let buildApiUrl (config : GoldMachineConfig) =
    sprintf
      "%s/fund_etf_hist_em?symbol=%s&start_date=%s"
      config.ApiBaseUrl
      config.Symbol
      config.StartDate
