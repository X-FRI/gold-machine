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
  /// Generates confidence-weighted trading signals based on prediction error.
  /// </summary>
  /// <param name="predictedPrice">Predicted price for next period.</param>
  /// <param name="actualPrice">Current actual price.</param>
  /// <param name="currentMAE">Current Mean Absolute Error of the model.</param>
  /// <returns>WeightedSignal containing signal strength and confidence.</returns>
  let generateWeightedSignal
    (predictedPrice : float32)
    (actualPrice : float)
    (currentMAE : float32)
    =
    let error = abs (float predictedPrice - actualPrice)
    let confidence = max 0.0 (1.0 - (error / float currentMAE))
    let rawSignal = if float predictedPrice > actualPrice then 1.0f else -1.0f
    let riskAdjustedSignal = rawSignal * float32 confidence

    { Signal = rawSignal
      Confidence = confidence
      RiskAdjustedSignal = riskAdjustedSignal }

  /// <summary>
  /// Implements multi-threshold trading strategy based on MAE levels.
  /// </summary>
  /// <param name="weightedSignal">The weighted signal from the model.</param>
  /// <param name="mae">Current Mean Absolute Error.</param>
  /// <returns>Position size recommendation (0.0 to 1.0).</returns>
  let calculateMultiThresholdPosition
    (weightedSignal : WeightedSignal)
    (mae : float32)
    =
    let maeFloat = float mae

    if maeFloat < 0.02 then
      // MAE < 0.02: full position buy
      if weightedSignal.RiskAdjustedSignal > 0.0f then 1.0f else 0.0f
    elif maeFloat < 0.05 then
      // 0.02 ≤ MAE < 0.05: half position buy
      if weightedSignal.RiskAdjustedSignal > 0.0f then 0.5f else 0.0f
    else
      // MAE ≥ 0.05: hold
      0.0f

  /// <summary>
  /// Implements adaptive trading strategy that adjusts based on market conditions.
  /// </summary>
  /// <param name="weightedSignal">The weighted signal from the model.</param>
  /// <param name="mae">Current Mean Absolute Error.</param>
  /// <param name="mape">Current Mean Absolute Percentage Error.</param>
  /// <param name="rmse">Current Root Mean Squared Error.</param>
  /// <param name="volatility">Current market volatility (price changes std dev).</param>
  /// <returns>Adaptive position size and leverage adjustment.</returns>
  let calculateAdaptivePosition
    (weightedSignal : WeightedSignal)
    (mae : float32)
    (mape : float)
    (rmse : float32)
    (volatility : float)
    =
    let basePosition = weightedSignal.RiskAdjustedSignal

    // MAPE increasing reduces leverage
    let mapeAdjustment = max 0.1 (1.0 - mape / 10.0)

    // RMSE too large stops trading
    let rmseThreshold = float (mae * 2.5f)
    let rmseAdjustment = if float rmse > rmseThreshold then 0.0 else 1.0

    // Volatility adjustment
    let volatilityAdjustment = max 0.2 (1.0 - volatility / 0.05)

    // MAE continuous decrease increases position
    let maeAdjustment = min 1.5 (1.0 + (0.05 - float mae) / 0.05)

    let finalPosition =
      float basePosition
      * mapeAdjustment
      * rmseAdjustment
      * volatilityAdjustment
      * maeAdjustment

    let clampedPosition = max 0.0 (min 1.0 finalPosition)

    {| Position = clampedPosition
       MapeAdjustment = mapeAdjustment
       RmseAdjustment = rmseAdjustment
       VolatilityAdjustment = volatilityAdjustment
       MaeAdjustment = maeAdjustment |}

  /// <summary>
  /// Calculates market volatility based on recent price changes.
  /// </summary>
  /// <param name="priceChanges">Array of recent price changes.</param>
  /// <returns>Market volatility measure.</returns>
  let calculateMarketVolatility (priceChanges : float[]) =
    if priceChanges.Length = 0 then
      0.0
    else
      let mean = Array.average priceChanges

      let variance =
        priceChanges |> Array.averageBy (fun x -> (x - mean) ** 2.0)

      sqrt variance

  /// <summary>
  /// Classifies current market regime based on volatility and model errors.
  /// </summary>
  /// <param name="volatility">Current market volatility.</param>
  /// <param name="mape">Current MAPE.</param>
  /// <returns>MarketRegime classification.</returns>
  let classifyMarketRegime (volatility : float) (mape : float) =
    match volatility, mape with
    | v, m when v < 0.01 && m < 1.0 -> LowVolatility
    | v, m when v < 0.03 && m < 3.0 -> NormalVolatility
    | v, m when v < 0.05 && m < 5.0 -> HighVolatility
    | _ -> ExtremeVolatility

  /// <summary>
  /// Generates advanced trading signals with confidence weighting and position sizing.
  /// </summary>
  /// <param name="predictedPrices">Array of predicted prices.</param>
  /// <param name="actualPrices">Array of actual prices.</param>
  /// <param name="evaluation">Model evaluation metrics.</param>
  /// <returns>Tuple of (weightedSignals, positionSizes, marketRegime).</returns>
  let generateAdvancedSignals
    (predictedPrices : float32[])
    (actualPrices : float32[])
    (evaluation : ModelEvaluation)
    =
    match
      DataProcessing.validateArrayLengths [| predictedPrices ; actualPrices |]
    with
    | Error _ -> [||], [||], NormalVolatility
    | Ok _ ->
      let priceChanges =
        actualPrices
        |> Array.map float
        |> DataProcessing.calculatePercentageChange

      let volatility = calculateMarketVolatility priceChanges
      let regime = classifyMarketRegime volatility evaluation.MAPE

      let weightedSignals =
        Array.zip predictedPrices actualPrices
        |> Array.map (fun (pred, actual) ->
          generateWeightedSignal pred (float actual) evaluation.MAE)

      let positionSizes
        : {| Position : float
             MapeAdjustment : float
             RmseAdjustment : float
             VolatilityAdjustment : float
             MaeAdjustment : float |}[] =
        weightedSignals
        |> Array.map (fun ws ->
          calculateAdaptivePosition
            ws
            evaluation.MAE
            evaluation.MAPE
            evaluation.RMSE
            volatility)

      weightedSignals,
      (positionSizes |> Array.map (fun ps -> ps.Position)),
      regime

  /// <summary>
  /// Calculates optimal position size using the Kelly Criterion.
  /// </summary>
  /// <param name="expectedReturn">Expected return of the strategy.</param>
  /// <param name="riskFreeRate">Risk-free rate.</param>
  /// <param name="volatility">Strategy volatility (standard deviation).</param>
  /// <returns>Kelly optimal position size (0.0 to 1.0).</returns>
  let calculateKellyPosition
    (expectedReturn : float)
    (riskFreeRate : float)
    (volatility : float)
    =
    if volatility = 0.0 then
      0.0
    else
      let riskPremium = expectedReturn - riskFreeRate
      let kellyFraction = riskPremium / (volatility ** 2.0)
      max 0.0 (min 1.0 kellyFraction) // Limit between 0 and 1

  /// <summary>
  /// Calculates position sizing using MAE as volatility proxy.
  /// </summary>
  /// <param name="expectedReturn">Expected return.</param>
  /// <param name="riskFreeRate">Risk-free rate.</param>
  /// <param name="mae">Mean Absolute Error as volatility measure.</param>
  /// <param name="currentPrice">Current price for scaling.</param>
  /// <returns>Position sizing recommendations.</returns>
  let calculatePositionSizingMAE
    (expectedReturn : float)
    (riskFreeRate : float)
    (mae : float32)
    (currentPrice : float)
    =
    // Use MAE as a volatility proxy
    let volatility = float mae / currentPrice // Relative volatility

    let kellyFraction =
      calculateKellyPosition expectedReturn riskFreeRate volatility

    // Calculate maximum drawdown (using MAE as a multiple)
    let maxDrawdown = min 0.5 (float mae * 3.0 / currentPrice)

    { PositionSize = kellyFraction
      MaxDrawdown = maxDrawdown
      KellyFraction = kellyFraction }

  /// <summary>
  /// Implements dynamic stop-loss based on RMSE.
  /// </summary>
  /// <param name="entryPrice">Entry price.</param>
  /// <param name="rmse">Root Mean Squared Error.</param>
  /// <param name="volatilityMultiplier">Multiplier for volatility adjustment.</param>
  /// <returns>Stop-loss price.</returns>
  let calculateDynamicStopLoss
    (entryPrice : float)
    (rmse : float32)
    (volatilityMultiplier : float)
    =
    let stopDistance = float rmse * volatilityMultiplier
    entryPrice - stopDistance

  /// <summary>
  /// Calculates trading costs including slippage.
  /// </summary>
  /// <param name="price">Transaction price.</param>
  /// <param name="quantity">Quantity traded.</param>
  /// <param name="commissionRate">Commission rate per trade.</param>
  /// <param name="mae">MAE for slippage estimation.</param>
  /// <returns>Total transaction cost.</returns>
  let calculateTradingCost
    (price : float)
    (quantity : float)
    (commissionRate : float)
    (mae : float32)
    =
    let notionalValue = price * quantity
    let commission = notionalValue * commissionRate

    // Use MAE to estimate slippage cost
    let slippage = float mae * quantity

    commission + slippage

  /// <summary>
  /// Evaluates strategy performance considering trading costs.
  /// </summary>
  /// <param name="strategyReturns">Array of strategy returns.</param>
  /// <param name="tradingCosts">Array of trading costs per period.</param>
  /// <returns>Net performance metrics.</returns>
  let evaluateStrategyWithCosts
    (strategyReturns : float[])
    (tradingCosts : float[])
    =
    if strategyReturns.Length <> tradingCosts.Length then
      failwith "Strategy returns and trading costs must have the same length"

    let netReturns =
      Array.zip strategyReturns tradingCosts
      |> Array.map (fun (ret, cost) -> ret - cost)

    let totalReturn = Array.sum netReturns
    let annualizedReturn = totalReturn * 252.0 / float strategyReturns.Length // 假设252个交易日

    let volatility = DataProcessing.calculateVolatility netReturns
    let sharpeRatio = DataProcessing.calculateSharpeRatio netReturns 0.02 // 2%无风险利率

    {| TotalReturn = totalReturn
       AnnualizedReturn = annualizedReturn
       Volatility = volatility
       SharpeRatio = sharpeRatio
       MaxDrawdown = 0.0 |} // Can implement more complex minimum drawdown calculation

  /// <summary>
  /// Monitors data quality and detects anomalies.
  /// </summary>
  /// <param name="recentPrices">Recent price observations.</param>
  /// <param name="baselineMAPE">Baseline MAPE for comparison.</param>
  /// <returns>Data quality assessment.</returns>
  let monitorDataQuality (recentPrices : float[]) (baselineMAPE : float) =
    if recentPrices.Length < 2 then
      {| IsAnomalous = false
         Message = "Insufficient data for quality check" |}
    else
      let priceChanges = DataProcessing.calculatePercentageChange recentPrices
      let currentVolatility = calculateMarketVolatility priceChanges

      // Calculate current MAPE proxy (simplified version)
      let avgChange = Array.average (priceChanges |> Array.map abs)
      let currentMAPE = avgChange * 100.0

      let mapeIncrease = currentMAPE - baselineMAPE
      let isAnomalous = mapeIncrease > 1.0 // MAPE suddenly increases

      let message =
        if isAnomalous then
          sprintf
            "Data quality anomaly: MAPE increased %.2f%%, possibly due to data source issues"
            mapeIncrease
        else
          "Data quality normal"

      {| IsAnomalous = isAnomalous
         Message = message |}

  /// <summary>
  /// Adapts strategy parameters based on market regime.
  /// </summary>
  /// <param name="regime">Current market regime.</param>
  /// <param name="baseParameters">Base strategy parameters.</param>
  /// <returns>Adapted strategy parameters.</returns>
  let adaptStrategyToRegime
    (regime : MarketRegime)
    (baseParameters : Map<string, float>)
    =
    let adjustments =
      match regime with
      | LowVolatility ->
        // Low volatility period: increase position, use MAE optimization
        Map
          [ "positionMultiplier", 1.2
            "stopLossMultiplier", 1.5
            "confidenceThreshold", 0.6 ]
      | NormalVolatility ->
        // Normal volatility period: standard parameters
        Map
          [ "positionMultiplier", 1.0
            "stopLossMultiplier", 1.0
            "confidenceThreshold", 0.7 ]
      | HighVolatility ->
        // High volatility period: reduce position, use RMSE priority
        Map
          [ "positionMultiplier", 0.7
            "stopLossMultiplier", 0.8
            "confidenceThreshold", 0.8 ]
      | ExtremeVolatility ->
        // Extreme volatility period: minimum position, strict stop-loss
        Map
          [ "positionMultiplier", 0.3
            "stopLossMultiplier", 0.5
            "confidenceThreshold", 0.9 ]

    // Merge base parameters and adjustment parameters
    baseParameters
    |> Map.fold
      (fun acc k v ->
        match Map.tryFind k adjustments with
        | Some adjustment -> Map.add k (v * adjustment) acc
        | None -> Map.add k v acc)
      Map.empty

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
