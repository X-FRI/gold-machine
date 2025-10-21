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
      RiskFreeRate = 0.0 } // Risk-free rate for Sharpe ratio calculation

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
