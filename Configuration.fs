namespace GoldMachine

open System

/// <summary>
/// Configuration management for the Gold Price Prediction System.
/// Provides default settings and validation for system parameters.
/// </summary>
module Configuration =

  /// <summary>
  /// Gets the default configuration for the gold price prediction system.
  /// </summary>
  /// <returns>A GoldMachineConfig with default values.</returns>
  let getDefaultConfig () =
    { ApiBaseUrl = "http://127.0.0.1:8080/api/public"
      Symbol = "518880" // GLD ETF symbol
      StartDate = "20000101" // Start from year 2000
      TrainRatio = 0.8 // 80% training, 20% testing
      RiskFreeRate = 0.0 // Risk-free rate for Sharpe ratio calculation
      DataProvider = ETFProvider }

  /// <summary>
  /// Gets the configuration for Shanghai Gold Exchange data.
  /// </summary>
  /// <returns>A GoldMachineConfig configured for SGE data.</returns>
  let getSGEConfig () =
    { ApiBaseUrl = "http://127.0.0.1:8080/api/public"
      Symbol = "Au99.99" // SGE gold symbol
      StartDate = "20000101" // Start from year 2000
      TrainRatio = 0.8 // 80% training, 20% testing
      RiskFreeRate = 0.0 // Risk-free rate for Sharpe ratio calculation
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
      { ApiBaseUrl = "http://127.0.0.1:8080/api/public"
        Symbol = etfSymbol // Custom ETF symbol
        StartDate = "20000101" // Start from year 2000
        TrainRatio = 0.8 // 80% training, 20% testing
        RiskFreeRate = 0.0 // Risk-free rate for Sharpe ratio calculation
        DataProvider = ETFProvider }

  /// <summary>
  /// Validates the configuration parameters.
  /// </summary>
  /// <param name="config">The configuration to validate.</param>
  /// <returns>Result indicating success or validation error.</returns>
  let validateConfig (config : GoldMachineConfig) =
    if String.IsNullOrWhiteSpace (config.ApiBaseUrl) then
      Error (ConfigurationError "API base URL cannot be empty")
    elif String.IsNullOrWhiteSpace (config.Symbol) then
      Error (ConfigurationError "Symbol cannot be empty")
    elif String.IsNullOrWhiteSpace (config.StartDate) then
      Error (ConfigurationError "Start date cannot be empty")
    elif config.TrainRatio <= 0.0 || config.TrainRatio >= 1.0 then
      Error (ConfigurationError "Train ratio must be between 0 and 1")
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
