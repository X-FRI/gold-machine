namespace GoldMachine

/// <summary>
/// Trading strategy module for generating trading signals and calculating strategy performance.
/// Implements a simple momentum-based strategy using price predictions.
/// </summary>
module TradingStrategy =

  /// <summary>
  /// Simple logging function for trading strategy operations.
  /// </summary>
  let logInfo (message : string) =
    printfn
      "[%s] INFO: %s"
      (System.DateTime.Now.ToString ("yyyy-MM-dd HH:mm:ss"))
      message

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
  /// Generates simple trading signals based on predicted vs actual prices.
  /// </summary>
  /// <param name="predictedPrices">Array of predicted prices.</param>
  /// <param name="actualPrices">Array of actual prices.</param>
  /// <returns>Array of trading signals (1.0 for buy, -1.0 for sell, 0.0 for hold).</returns>
  let generateSimpleSignals
    (predictedPrices : float32[])
    (actualPrices : float32[])
    =
    match
      DataProcessing.validateArrayLengths [| predictedPrices ; actualPrices |]
    with
    | Error _ -> [||]
    | Ok _ ->
      Array.zip predictedPrices actualPrices
      |> Array.map (fun (pred, actual) ->
        if float pred > float actual then 1.0
        elif float pred < float actual then -1.0
        else 0.0)

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

    // Generate signals based on predicted vs previous actual prices (more realistic)
    let signals =
      if actualPrices.Length < 2 then
        [||]
      else
        Array.zip predictedPrices.[..actualPrices.Length-2] actualPrices.[..actualPrices.Length-2]
        |> Array.map (fun (pred, prevActual) ->
          if float pred > float prevActual then 1.0
          elif float pred < float prevActual then -1.0
          else 0.0)

    let strategyReturns = calculateStrategyReturns priceChanges signals
    let cumulativeReturns = calculateCumulativeStrategyReturns strategyReturns

    let sharpeRatio =
      DataProcessing.calculateSharpeRatio strategyReturns config.RiskFreeRate

    (signals, strategyReturns, cumulativeReturns, sharpeRatio)

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

    // Calculate maximum drawdown
    let cumulativeReturns = DataProcessing.calculateCumulativeReturns strategyReturns
    let mutable maxDrawdown = 0.0
    let mutable peakValue = 0.0

    for ret in cumulativeReturns do
      if ret > peakValue then
        peakValue <- ret
      else
        let currentDrawdown = peakValue - ret
        maxDrawdown <- max maxDrawdown currentDrawdown

    // Calculate win rate (percentage of positive returns)
    let winningDays = strategyReturns |> Array.filter (fun r -> r > 0.0)
    let winRate = float winningDays.Length / float strategyReturns.Length

    {| TotalReturn = totalReturn
       SharpeRatio = sharpeRatio
       MaxDrawdown = maxDrawdown
       WinRate = winRate |}

  /// <summary>
  /// Represents the result of a backtesting simulation.
  /// </summary>
  type BacktestResult =
    { TotalReturn : float
      AnnualizedReturn : float
      Volatility : float
      SharpeRatio : float
      MaxDrawdown : float
      WinRate : float
      TotalTrades : int
      ProfitFactor : float }

  /// <summary>
  /// Performs walk-forward backtesting with expanding window.
  /// </summary>
  /// <param name="historicalData">Complete historical dataset.</param>
  /// <param name="initialTrainSize">Initial training window size.</param>
  /// <param name="testWindowSize">Size of each test window.</param>
  /// <param name="trainModel">Function to train model on training data.</param>
  /// <param name="config">Configuration parameters.</param>
  /// <returns>BacktestResult with comprehensive performance metrics.</returns>
  let performWalkForwardBacktest
    (historicalData : GoldDataRecord[])
    (initialTrainSize : int)
    (testWindowSize : int)
    (trainModel : GoldDataRecord[] -> 'T)
    (predictFunc : 'T -> GoldDataRecord -> float32)
    (config : GoldMachineConfig)
    =
    if historicalData.Length < initialTrainSize + testWindowSize then
      failwith "Insufficient data for walk-forward backtesting"

    let mutable currentTrainSize = initialTrainSize
    let mutable allReturns = []
    let mutable allTrades = [] // List of (entryPrice, exitPrice, direction, profit)
    let mutable peakValue = 0.0 // Start from 0 for cumulative returns

    while currentTrainSize + testWindowSize <= historicalData.Length do
      // Split data into training and testing windows
      let trainData = historicalData.[.. currentTrainSize - 1]

      let testData =
        historicalData.[currentTrainSize .. currentTrainSize + testWindowSize
                                            - 1]

      // Train model on current training window
      let model = trainModel trainData

      // Generate predictions for test window
      let predictions =
        testData
        |> Array.map (fun record -> predictFunc model record)

      // Generate signals and simulate trading
      let actualPrices = testData |> Array.map (fun r -> float32 r.Close)
      let signals = generateSimpleSignals predictions actualPrices

      // Simulate trading for this window
      let mutable currentPosition = 0.0 // 0 = no position, 1 = long, -1 = short
      let mutable entryPrice = 0.0

      for i in 0 .. signals.Length - 1 do
        let signal = signals.[i]
        let currentPrice = float testData.[i].Close

        match currentPosition, signal with
        // No position - check for entry signals
        | 0.0, s when s > 0.0 ->
          // Enter long position
          currentPosition <- 1.0
          entryPrice <- currentPrice
        | 0.0, s when s < 0.0 ->
          // Enter short position
          currentPosition <- -1.0
          entryPrice <- currentPrice
        // Have position - check for exit signals
        | pos, s when (pos > 0.0 && s <= 0.0) || (pos < 0.0 && s >= 0.0) ->
          // Exit position
          let exitPrice = currentPrice
          let profit = (exitPrice - entryPrice) * pos // pos is +/- 1
          let trade = (entryPrice, exitPrice, pos, profit)
          allTrades <- trade :: allTrades
          currentPosition <- 0.0
        | _ -> () // Hold position

        // Calculate daily return if in position
        if currentPosition <> 0.0 then
          let dailyReturn =
            if i > 0 then
              let prevPrice = float testData.[i - 1].Close
              (currentPrice - prevPrice) / prevPrice * currentPosition
            else
              0.0 // No return on entry day
          allReturns <- dailyReturn :: allReturns

      // If still in position at end of window, close it
      if currentPosition <> 0.0 then
        let exitPrice = float testData.[testData.Length - 1].Close
        let profit = (exitPrice - entryPrice) * currentPosition
        let trade = (entryPrice, exitPrice, currentPosition, profit)
        allTrades <- trade :: allTrades

      // Expand training window for next iteration
      currentTrainSize <- currentTrainSize + testWindowSize

    // Reverse lists since we prepended
    let allReturns = List.rev allReturns
    let allTrades = List.rev allTrades

    // Calculate performance metrics
    let totalReturn = allReturns |> List.sum

    // Annualized return (252 trading days per year)
    let annualizedReturn =
      if allReturns.Length > 0 then
        totalReturn * 252.0 / float allReturns.Length
      else
        0.0

    let volatility =
      DataProcessing.calculateVolatility (allReturns |> List.toArray)

    let sharpeRatio =
      DataProcessing.calculateSharpeRatio
        (allReturns |> List.toArray)
        config.RiskFreeRate

    // Calculate maximum drawdown
    let cumulativeReturns =
      DataProcessing.calculateCumulativeReturns (allReturns |> List.toArray)

    let mutable maxDrawdown = 0.0
    let mutable peakValue = 0.0

    for ret in cumulativeReturns do
      if ret > peakValue then
        peakValue <- ret
      else
        let currentDrawdown = peakValue - ret
        maxDrawdown <- max maxDrawdown currentDrawdown

    // Calculate win rate and profit factor from actual trades
    let winningTrades = allTrades |> List.filter (fun (_, _, _, profit) -> profit > 0.0)
    let losingTrades = allTrades |> List.filter (fun (_, _, _, profit) -> profit < 0.0)

    let winRate =
      if allTrades.Length > 0 then
        float winningTrades.Length / float allTrades.Length
      else
        0.0

    // Calculate profit factor (gross profit / gross loss)
    let grossProfit = winningTrades |> List.sumBy (fun (_, _, _, profit) -> profit)
    let grossLoss = losingTrades |> List.sumBy (fun (_, _, _, profit) -> abs profit)

    let profitFactor =
      if grossLoss > 0.0 then grossProfit / grossLoss
      else if grossProfit > 0.0 then 999.0
      else 1.0

    { TotalReturn = totalReturn
      AnnualizedReturn = annualizedReturn
      Volatility = volatility
      SharpeRatio = sharpeRatio
      MaxDrawdown = maxDrawdown
      WinRate = winRate
      TotalTrades = allTrades.Length
      ProfitFactor = profitFactor }

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
