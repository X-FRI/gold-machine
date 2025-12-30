namespace GoldMachine

/// <summary>
/// Trading strategy module for generating trading signals and calculating strategy performance.
/// Implements a simple momentum-based strategy using price predictions with ATR-based risk management.
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
  /// Represents a position with ATR-based stop loss and take profit levels.
  /// </summary>
  type Position =
    { EntryPrice : float
      EntryIndex : int
      Direction : float // 1.0 for long, -1.0 for short
      StopLoss : float
      TakeProfit : float
      TrailingStop : float option
      PositionSize : float }

  /// <summary>
  /// Calculates ATR-based stop loss price.
  /// </summary>
  /// <param name="entryPrice">Entry price of the position.</param>
  /// <param name="atr">Current ATR value.</param>
  /// <param name="direction">Direction: 1.0 for long, -1.0 for short.</param>
  /// <param name="multiplier">ATR multiplier for stop loss distance.</param>
  /// <returns>Stop loss price.</returns>
  let calculateATRStopLoss
    (entryPrice : float)
    (atr : float32)
    (direction : float)
    (multiplier : float)
    =
    let stopDistance = float atr * multiplier

    if direction > 0.0 then
      // Long position: stop loss below entry
      entryPrice - stopDistance
    else
      // Short position: stop loss above entry
      entryPrice + stopDistance

  /// <summary>
  /// Calculates ATR-based take profit price.
  /// </summary>
  /// <param name="entryPrice">Entry price of the position.</param>
  /// <param name="atr">Current ATR value.</param>
  /// <param name="direction">Direction: 1.0 for long, -1.0 for short.</param>
  /// <param name="multiplier">ATR multiplier for take profit distance.</param>
  /// <returns>Take profit price.</returns>
  let calculateATRTakeProfit
    (entryPrice : float)
    (atr : float32)
    (direction : float)
    (multiplier : float)
    =
    let profitDistance = float atr * multiplier

    if direction > 0.0 then
      // Long position: take profit above entry
      entryPrice + profitDistance
    else
      // Short position: take profit below entry
      entryPrice - profitDistance

  /// <summary>
  /// Calculates ATR-based position size based on current volatility.
  /// </summary>
  /// <param name="currentATR">Current ATR value.</param>
  /// <param name="baselineATR">Baseline ATR (average over baseline period).</param>
  /// <param name="basePositionSize">Base position size (e.g., 0.2 = 20%).</param>
  /// <param name="maxPositionSize">Maximum position size (e.g., 0.3 = 30%).</param>
  /// <param name="minPositionSize">Minimum position size (e.g., 0.05 = 5%).</param>
  /// <returns>Adjusted position size as percentage of capital.</returns>
  let calculateATRPositionSize
    (currentATR : float32)
    (baselineATR : float32)
    (basePositionSize : float)
    (maxPositionSize : float)
    (minPositionSize : float)
    =
    if baselineATR = 0.0f then
      basePositionSize
    else
      // Calculate adjustment factor: lower ATR = larger position, higher ATR = smaller position
      let adjustmentFactor = float baselineATR / float currentATR
      let adjustedSize = basePositionSize * adjustmentFactor

      // Clamp to min/max bounds
      max minPositionSize (min maxPositionSize adjustedSize)

  /// <summary>
  /// Updates trailing stop loss for a position.
  /// </summary>
  /// <param name="position">Current position.</param>
  /// <param name="currentPrice">Current market price.</param>
  /// <param name="atr">Current ATR value.</param>
  /// <param name="multiplier">ATR multiplier for trailing stop distance.</param>
  /// <returns>Updated position with new trailing stop.</returns>
  let updateTrailingStop
    (position : Position)
    (currentPrice : float)
    (atr : float32)
    (multiplier : float)
    =
    if position.Direction > 0.0 then
      // Long position: trailing stop moves up only
      let newTrailingStop = currentPrice - (float atr * multiplier)

      { position with
          TrailingStop =
            match position.TrailingStop with
            | Some existingStop -> Some(max existingStop newTrailingStop)
            | None -> Some newTrailingStop }
    else
      // Short position: trailing stop moves down only
      let newTrailingStop = currentPrice + (float atr * multiplier)

      { position with
          TrailingStop =
            match position.TrailingStop with
            | Some existingStop -> Some(min existingStop newTrailingStop)
            | None -> Some newTrailingStop }

  /// <summary>
  /// Calculates baseline ATR (average ATR over a period).
  /// </summary>
  /// <param name="atrValues">Array of ATR values.</param>
  /// <param name="period">Period for baseline calculation.</param>
  /// <param name="index">Current index.</param>
  /// <returns>Baseline ATR value.</returns>
  let calculateBaselineATR
    (atrValues : float32[])
    (period : int)
    (index : int)
    =
    if atrValues.Length = 0 then
      0.0f
    else
      let startIndex = max 0 (index - period + 1)
      let endIndex = min (atrValues.Length - 1) index
      let window = atrValues.[startIndex .. endIndex]

      if window.Length > 0 then
        window |> Array.average
      else
        atrValues.[index]

  /// <summary>
  /// Generates trading signals based on predicted vs actual prices with risk management.
  /// Returns 1.0 for buy signals, -1.0 for sell signals, and 0.0 for hold signals.
  /// Includes basic risk management rules.
  /// </summary>
  /// <param name="predictedPrices">Array of predicted prices.</param>
  /// <param name="currentPrices">Array of actual current prices.</param>
  /// <param name="threshold">Minimum price difference threshold for signals (default 0.001).</param>
  /// <returns>Array of trading signals.</returns>
  let generateTradingSignals
    (predictedPrices : float32[])
    (currentPrices : float32[])
    (threshold : float)
    =
    match
      DataProcessing.validateArrayLengths [| predictedPrices ; currentPrices |]
    with
    | Error _ -> [||]
    | Ok _ ->
      Array.zip predictedPrices currentPrices
      |> Array.map (fun (pred, curr) ->
        let priceDiff = float (pred - curr) / float curr

        if priceDiff > threshold then 1.0 // Buy signal
        elif priceDiff < -threshold then -1.0 // Sell signal
        else 0.0) // Hold signal

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

    // Generate enhanced signals with risk management
    let signals =
      if actualPrices.Length < 2 then
        [||]
      else
        // Use predicted vs previous actual prices for more realistic signals
        let signalPrices = actualPrices.[.. actualPrices.Length - 2]
        let signalPredictions = predictedPrices.[.. actualPrices.Length - 2]

        // Use adaptive threshold based on recent volatility
        let recentVolatility =
          if signalPrices.Length >= 10 then
            let recentPrices = signalPrices.[signalPrices.Length - 10 ..]

            let returns =
              DataProcessing.calculatePercentageChange (
                recentPrices |> Array.map float
              )

            DataProcessing.calculateVolatility returns
          else
            0.01 // Default 1% threshold

        let threshold = max 0.001 recentVolatility // Minimum 0.1%, maximum based on volatility

        generateTradingSignals signalPredictions signalPrices threshold

    let strategyReturns = calculateStrategyReturns priceChanges signals
    let cumulativeReturns = calculateCumulativeStrategyReturns strategyReturns

    let sharpeRatio =
      DataProcessing.calculateSharpeRatio strategyReturns config.RiskFreeRate

    (signals, strategyReturns, cumulativeReturns, sharpeRatio)

  /// <summary>
  /// Generates a trading recommendation based on the latest prediction with enhanced signals.
  /// </summary>
  /// <param name="currentPrice">Current market price.</param>
  /// <param name="predictedPrice">Predicted price for next period.</param>
  /// <param name="threshold">Price difference threshold for signals.</param>
  /// <returns>String describing the trading signal.</returns>
  let generateTradingRecommendation
    (currentPrice : float)
    (predictedPrice : float32)
    (threshold : float)
    =
    let priceDiff = (float predictedPrice - currentPrice) / currentPrice

    if priceDiff > threshold then
      sprintf
        "BUY GLD - Predicted price %.2f is %.2f%% higher than current price %.2f"
        (float predictedPrice)
        (priceDiff * 100.0)
        currentPrice
    elif priceDiff < -threshold then
      sprintf
        "SELL GLD - Predicted price %.2f is %.2f%% lower than current price %.2f"
        (float predictedPrice)
        (abs (priceDiff) * 100.0)
        currentPrice
    else
      sprintf
        "HOLD GLD - Predicted price %.2f is within %.2f%% of current price %.2f"
        (float predictedPrice)
        (threshold * 100.0)
        currentPrice

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
    let cumulativeReturns =
      DataProcessing.calculateCumulativeReturns strategyReturns

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
  /// Performs walk-forward backtesting with expanding window and ATR-based risk management.
  /// </summary>
  /// <param name="historicalData">Complete historical dataset.</param>
  /// <param name="initialTrainSize">Initial training window size.</param>
  /// <param name="testWindowSize">Size of each test window.</param>
  /// <param name="trainModel">Function to train model on training data.</param>
  /// <param name="predictFunc">Function to generate predictions from model.</param>
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

    let atrConfig = config.ATRRiskConfig
    
    // Log ATR configuration
    logInfo (
      sprintf
        "ATR Risk Management: StopLoss=%.1fx ATR, TakeProfit=%.1fx ATR, PositionSizing=%s, TrailingStop=%s"
        atrConfig.StopLossMultiplier
        atrConfig.TakeProfitMultiplier
        (if atrConfig.PositionSizingEnabled then "Enabled" else "Disabled")
        (if atrConfig.TrailingStopEnabled then "Enabled" else "Disabled")
    )
    
    let mutable currentTrainSize = initialTrainSize
    let mutable allReturns = []
    let mutable allTrades = [] // List of (entryPrice, exitPrice, direction, profit, positionSize)
    let mutable peakValue = 0.0 // Start from 0 for cumulative returns
    let mutable stopLossTriggers = 0
    let mutable takeProfitTriggers = 0
    let mutable trailingStopTriggers = 0
    let mutable signalReversalExits = 0

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
        testData |> Array.map (fun record -> predictFunc model record)

      // Generate signals and simulate trading
      let actualPrices = testData |> Array.map (fun r -> float32 r.Close)
      let signals = generateSimpleSignals predictions actualPrices
      let atrValues = testData |> Array.map (fun r -> r.ATR)
      
      // Validate ATR values
      let validATRCount = atrValues |> Array.filter (fun atr -> atr > 0.0f) |> Array.length
      if validATRCount = 0 && testData.Length > 0 then
        logInfo (
          sprintf
            "Warning: No valid ATR values found in test window (window size: %d). ATR-based risk management may not work correctly."
            testData.Length
        )

      // Calculate baseline ATR for position sizing
      let baselineATR =
        if atrValues.Length >= atrConfig.ATRBaselinePeriod then
          let baselineWindow =
            atrValues.[.. atrConfig.ATRBaselinePeriod - 1]

          baselineWindow |> Array.average
        else if atrValues.Length > 0 then
          atrValues |> Array.average
        else
          0.0f
      
      // Log ATR information for first window only
      if currentTrainSize = initialTrainSize then
        let avgATR = if atrValues.Length > 0 then atrValues |> Array.average else 0.0f
        logInfo (
          sprintf
            "ATR Data: Window size=%d, Valid ATR values=%d, Average ATR=%.4f, Baseline ATR=%.4f"
            testData.Length
            validATRCount
            avgATR
            baselineATR
        )

      // Simulate trading for this window with ATR risk management
      let mutable currentPosition : Position option = None

      for i in 0 .. signals.Length - 1 do
        let signal = signals.[i]
        let currentPrice = float testData.[i].Close
        let currentATR = atrValues.[i]

        match currentPosition, signal with
        // No position - check for entry signals
        | None, s when s > 0.0 ->
          // Enter long position with ATR-based risk management
          let positionSize =
            if atrConfig.PositionSizingEnabled then
              calculateATRPositionSize
                currentATR
                baselineATR
                atrConfig.BasePositionSize
                atrConfig.MaxPositionSize
                atrConfig.MinPositionSize
            else
              atrConfig.BasePositionSize

          let stopLoss =
            calculateATRStopLoss
              currentPrice
              currentATR
              1.0
              atrConfig.StopLossMultiplier

          let takeProfit =
            calculateATRTakeProfit
              currentPrice
              currentATR
              1.0
              atrConfig.TakeProfitMultiplier

          currentPosition <-
            Some
              { EntryPrice = currentPrice
                EntryIndex = i
                Direction = 1.0
                StopLoss = stopLoss
                TakeProfit = takeProfit
                TrailingStop = None
                PositionSize = positionSize }

        | None, s when s < 0.0 ->
          // Enter short position with ATR-based risk management
          let positionSize =
            if atrConfig.PositionSizingEnabled then
              calculateATRPositionSize
                currentATR
                baselineATR
                atrConfig.BasePositionSize
                atrConfig.MaxPositionSize
                atrConfig.MinPositionSize
            else
              atrConfig.BasePositionSize

          let stopLoss =
            calculateATRStopLoss
              currentPrice
              currentATR
              -1.0
              atrConfig.StopLossMultiplier

          let takeProfit =
            calculateATRTakeProfit
              currentPrice
              currentATR
              -1.0
              atrConfig.TakeProfitMultiplier

          currentPosition <-
            Some
              { EntryPrice = currentPrice
                EntryIndex = i
                Direction = -1.0
                StopLoss = stopLoss
                TakeProfit = takeProfit
                TrailingStop = None
                PositionSize = positionSize }

        // Have position - check for exit signals (ATR stop loss, take profit, or signal reversal)
        | Some pos, _ ->
          let mutable shouldExit = false
          let mutable exitReason = ""

          // Check ATR stop loss
          if pos.Direction > 0.0 && currentPrice <= pos.StopLoss then
            shouldExit <- true
            exitReason <- "ATR Stop Loss"
          elif pos.Direction < 0.0 && currentPrice >= pos.StopLoss then
            shouldExit <- true
            exitReason <- "ATR Stop Loss"

          // Check ATR take profit
          if not shouldExit then
            if pos.Direction > 0.0 && currentPrice >= pos.TakeProfit then
              shouldExit <- true
              exitReason <- "ATR Take Profit"
            elif pos.Direction < 0.0 && currentPrice <= pos.TakeProfit then
              shouldExit <- true
              exitReason <- "ATR Take Profit"

          // Check trailing stop
          if not shouldExit && atrConfig.TrailingStopEnabled then
            let updatedPos = updateTrailingStop pos currentPrice currentATR atrConfig.StopLossMultiplier

            match updatedPos.TrailingStop with
            | Some trailingStop ->
              if pos.Direction > 0.0 && currentPrice <= trailingStop then
                shouldExit <- true
                exitReason <- "Trailing Stop"
              elif pos.Direction < 0.0 && currentPrice >= trailingStop then
                shouldExit <- true
                exitReason <- "Trailing Stop"

              currentPosition <- Some updatedPos
            | None -> currentPosition <- Some updatedPos

          // Check signal reversal
          if not shouldExit then
            if (pos.Direction > 0.0 && signal <= 0.0)
               || (pos.Direction < 0.0 && signal >= 0.0) then
              shouldExit <- true
              exitReason <- "Signal Reversal"

          // Exit position if needed
          if shouldExit then
            // Track exit reasons for statistics
            match exitReason with
            | "ATR Stop Loss" -> stopLossTriggers <- stopLossTriggers + 1
            | "ATR Take Profit" -> takeProfitTriggers <- takeProfitTriggers + 1
            | "Trailing Stop" -> trailingStopTriggers <- trailingStopTriggers + 1
            | "Signal Reversal" -> signalReversalExits <- signalReversalExits + 1
            | _ -> ()

            let exitPrice = currentPrice
            let profit =
              (exitPrice - pos.EntryPrice) * pos.Direction * pos.PositionSize

            let trade =
              (pos.EntryPrice, exitPrice, pos.Direction, profit, pos.PositionSize)

            allTrades <- trade :: allTrades
            currentPosition <- None
          else
            // Update trailing stop if enabled
            if atrConfig.TrailingStopEnabled then
              let updatedPos =
                updateTrailingStop pos currentPrice currentATR atrConfig.StopLossMultiplier

              currentPosition <- Some updatedPos

        | _ -> () // Hold position or no signal

        // Calculate daily return if in position
        match currentPosition with
        | Some pos ->
          let dailyReturn =
            if i > 0 then
              let prevPrice = float testData.[i - 1].Close
              (currentPrice - prevPrice)
              / prevPrice
              * pos.Direction
              * pos.PositionSize
            else
              0.0 // No return on entry day

          allReturns <- dailyReturn :: allReturns
        | None -> ()

      // If still in position at end of window, close it
      match currentPosition with
      | Some pos ->
        let exitPrice = float testData.[testData.Length - 1].Close
        let profit =
          (exitPrice - pos.EntryPrice) * pos.Direction * pos.PositionSize

        let trade =
          (pos.EntryPrice, exitPrice, pos.Direction, profit, pos.PositionSize)

        allTrades <- trade :: allTrades
      | None -> ()

      // Expand training window for next iteration
      currentTrainSize <- currentTrainSize + testWindowSize

    // Reverse lists since we prepended
    let allReturns = List.rev allReturns
    let allTrades = List.rev allTrades

    // Log ATR strategy statistics
    logInfo (
      sprintf
        "ATR Strategy Statistics: StopLoss=%d, TakeProfit=%d, TrailingStop=%d, SignalReversal=%d"
        stopLossTriggers
        takeProfitTriggers
        trailingStopTriggers
        signalReversalExits
    )

    // Calculate average position size if position sizing is enabled
    if atrConfig.PositionSizingEnabled && allTrades.Length > 0 then
      let avgPositionSize =
        allTrades
        |> List.map (fun (_, _, _, _, size) -> size)
        |> List.average

      let minPositionSize =
        allTrades
        |> List.map (fun (_, _, _, _, size) -> size)
        |> List.min

      let maxPositionSize =
        allTrades
        |> List.map (fun (_, _, _, _, size) -> size)
        |> List.max

      logInfo (
        sprintf
          "Position Sizing: Avg=%.2f%%, Min=%.2f%%, Max=%.2f%%"
          (avgPositionSize * 100.0)
          (minPositionSize * 100.0)
          (maxPositionSize * 100.0)
      )

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
    // Trade format: (entryPrice, exitPrice, direction, profit, positionSize)
    let winningTrades =
      allTrades |> List.filter (fun (_, _, _, profit, _) -> profit > 0.0)

    let losingTrades =
      allTrades |> List.filter (fun (_, _, _, profit, _) -> profit < 0.0)

    let winRate =
      if allTrades.Length > 0 then
        float winningTrades.Length / float allTrades.Length
      else
        0.0

    // Calculate profit factor (gross profit / gross loss)
    let grossProfit =
      winningTrades |> List.sumBy (fun (_, _, _, profit, _) -> profit)

    let grossLoss =
      losingTrades |> List.sumBy (fun (_, _, _, profit, _) -> abs profit)

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
