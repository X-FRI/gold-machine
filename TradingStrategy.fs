namespace GoldMachine

/// <summary>
/// Trading strategy module for generating trading signals and calculating strategy performance.
/// Implements a simple momentum-based strategy using price predictions.
/// </summary>
module TradingStrategy =

  /// <summary>
  /// Generates trading signals based on predicted vs actual prices.
  /// Returns 1.0 for buy signals (predicted > actual) and 0.0 for hold signals.
  /// </summary>
  /// <param name="predictedPrices">Array of predicted prices.</param>
  /// <param name="currentPrices">Array of actual current prices.</param>
  /// <returns>Array of trading signals.</returns>
  let generateTradingSignals
    (predictedPrices : float32[])
    (currentPrices : float32[])
    =
    match
      DataProcessing.validateArrayLengths [| predictedPrices ; currentPrices |]
    with
    | Error _ -> [||]
    | Ok _ ->
      Array.zip predictedPrices currentPrices
      |> Array.map (fun (pred, curr) -> if pred > curr then 1.0 else 0.0)

  /// <summary>
  /// Calculates strategy returns by multiplying price changes with trading signals.
  /// </summary>
  /// <param name="priceChanges">Array of price changes (returns).</param>
  /// <param name="signals">Array of trading signals.</param>
  /// <returns>Array of strategy returns.</returns>
  let calculateStrategyReturns (priceChanges : float[]) (signals : float[]) =
    match DataProcessing.validateArrayLengths [| priceChanges ; signals |] with
    | Error _ -> [||]
    | Ok _ ->
      Array.zip priceChanges signals
      |> Array.map (fun (change, signal) -> change * signal)

  /// <summary>
  /// Calculates cumulative returns for the trading strategy.
  /// </summary>
  /// <param name="returns">Array of periodic strategy returns.</param>
  /// <returns>Array of cumulative returns starting from 0.</returns>
  let calculateCumulativeStrategyReturns (returns : float[]) =
    DataProcessing.calculateCumulativeReturns returns

  /// <summary>
  /// Evaluates the complete trading strategy performance.
  /// </summary>
  /// <param name="predictedPrices">Predicted prices from the model.</param>
  /// <param name="actualPrices">Actual market prices.</param>
  /// <param name="config">Configuration containing risk-free rate.</param>
  /// <returns>Tuple of (signals, strategyReturns, cumulativeReturns, sharpeRatio).</returns>
  let evaluateStrategy
    (predictedPrices : float32[])
    (actualPrices : float32[])
    (config : GoldMachineConfig)
    =
    let priceChanges =
      actualPrices
      |> Array.map float
      |> DataProcessing.calculatePercentageChange

    let signals = generateTradingSignals predictedPrices actualPrices

    // Align signals with price changes (signals are one shorter due to differencing)
    let alignedSignals = signals.[.. priceChanges.Length - 1]
    let strategyReturns = calculateStrategyReturns priceChanges alignedSignals
    let cumulativeReturns = calculateCumulativeStrategyReturns strategyReturns

    let sharpeRatio =
      DataProcessing.calculateSharpeRatio strategyReturns config.RiskFreeRate

    (alignedSignals, strategyReturns, cumulativeReturns, sharpeRatio)

  /// <summary>
  /// Generates a trading recommendation based on the latest prediction.
  /// </summary>
  /// <param name="currentPrice">Current market price.</param>
  /// <param name="predictedPrice">Predicted price for next period.</param>
  /// <returns>String describing the trading signal.</returns>
  let generateTradingRecommendation
    (currentPrice : float)
    (predictedPrice : float32)
    =
    if float predictedPrice > currentPrice then
      "BUY GLD - Predicted price higher than current price"
    else
      "HOLD - Predicted price not higher than current price"

  /// <summary>
  /// Calculates strategy metrics for performance reporting.
  /// </summary>
  /// <param name="strategyReturns">Array of strategy returns.</param>
  /// <param name="config">Configuration with risk parameters.</param>
  /// <returns>Record containing strategy performance metrics.</returns>
  let calculateStrategyMetrics
    (strategyReturns : float[])
    (config : GoldMachineConfig)
    =
    let totalReturn = strategyReturns |> Array.sum

    let sharpeRatio =
      DataProcessing.calculateSharpeRatio strategyReturns config.RiskFreeRate

    let maxDrawdown = 0.0 // Could be implemented for more advanced analysis
    let winRate = 0.0 // Could be implemented for more advanced analysis

    {| TotalReturn = totalReturn
       SharpeRatio = sharpeRatio
       MaxDrawdown = maxDrawdown
       WinRate = winRate |}

  /// <summary>
  /// Validates trading strategy parameters.
  /// </summary>
  /// <param name="predictedPrices">Predicted prices array.</param>
  /// <param name="actualPrices">Actual prices array.</param>
  /// <returns>Result indicating validation success or error.</returns>
  let validateStrategyInputs
    (predictedPrices : float32[])
    (actualPrices : float32[])
    =
    if predictedPrices.Length = 0 then
      Error (ConfigurationError "Predicted prices array is empty")
    elif actualPrices.Length = 0 then
      Error (ConfigurationError "Actual prices array is empty")
    elif predictedPrices.Length <> actualPrices.Length then
      Error (
        ConfigurationError
          "Predicted and actual prices arrays must have the same length"
      )
    else
      Ok ()
