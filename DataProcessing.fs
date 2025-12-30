namespace GoldMachine

open System
open MathNet.Numerics.Statistics

/// <summary>
/// Data processing module for calculating technical indicators and statistical measures.
/// Provides functions for moving averages, returns calculations, and data transformations.
/// </summary>
module DataProcessing =

  /// <summary>
  /// Simple logging function for data processing operations.
  /// </summary>
  let logInfo (message : string) =
    printfn
      "[%s] INFO: %s"
      (DateTime.Now.ToString ("yyyy-MM-dd HH:mm:ss"))
      message

  /// <summary>
  /// Calculates simple moving average for the given window size.
  /// </summary>
  /// <param name="values">Array of price values.</param>
  /// <param name="windowSize">Size of the moving average window.</param>
  /// <returns>Result containing array of moving average values or error.</returns>
  let calculateMovingAverage (values : float[]) (windowSize : int) =
    if windowSize <= 0 then
      Error (DataAcquisitionFailed "Window size must be positive")
    elif values.Length < windowSize then
      Error (
        DataAcquisitionFailed
          $"Insufficient data: {values.Length} values, minimum {windowSize} required"
      )
    else
      Ok
        [| for i in windowSize - 1 .. values.Length - 1 do
             let window = values.[i - windowSize + 1 .. i]
             yield window |> Array.average |]

  /// <summary>
  /// Calculates percentage changes between consecutive price values.
  /// </summary>
  /// <param name="values">Array of price values.</param>
  /// <returns>Array of percentage changes.</returns>
  let calculatePercentageChange (values : float[]) =
    if values.Length < 2 then
      [||]
    else
      [| for i in 1 .. values.Length - 1 do
           yield (values.[i] - values.[i - 1]) / values.[i - 1] |]

  /// <summary>
  /// Calculates cumulative returns from an array of periodic returns.
  /// </summary>
  /// <param name="returns">Array of periodic returns.</param>
  /// <returns>Array of cumulative returns starting from 0.</returns>
  let calculateCumulativeReturns (returns : float[]) =
    returns |> Array.scan (+) 0.0

  /// <summary>
  /// Calculates the Sharpe ratio for a series of returns.
  /// </summary>
  /// <param name="returns">Array of periodic returns.</param>
  /// <param name="riskFreeRate">Risk-free rate for comparison.</param>
  /// <returns>The Sharpe ratio value.</returns>
  let calculateSharpeRatio (returns : float[]) (riskFreeRate : float) =
    if returns.Length = 0 then
      0.0
    else
      let meanReturn = Statistics.Mean (returns)
      let stdDev = Statistics.StandardDeviation (returns)

      if stdDev = 0.0 then 0.0 else (meanReturn - riskFreeRate) / stdDev

  /// <summary>
  /// Calculates the volatility (standard deviation) of a series of returns.
  /// </summary>
  /// <param name="returns">Array of periodic returns.</param>
  /// <returns>The volatility (standard deviation) value.</returns>
  let calculateVolatility (returns : float[]) =
    if returns.Length = 0 then 0.0 else Statistics.StandardDeviation (returns)

  /// <summary>
  /// Converts raw data source to standardized GoldDataRecord array.
  /// Handles different data formats from various providers.
  /// </summary>
  /// <param name="rawData">Raw data from any supported data source.</param>
  /// <returns>Array of standardized gold data records.</returns>
  let convertRawDataToRecords (rawData : RawDataSource) =
    match rawData with
    | ETF rawETFData ->
      rawETFData
      |> Array.filter (fun item ->
        not (String.IsNullOrWhiteSpace item.Date) && item.Close > 0.0)
      |> Array.map (fun item ->
        { Date = DateTime.Parse item.Date
          Open = item.Open
          High = item.High
          Low = item.Low
          Close = item.Close
          Volume = item.Volume
          Amount = item.Amount
          // Moving Averages (will be calculated later)
          MA3 = 0.0f
          MA5 = 0.0f
          MA9 = 0.0f
          MA20 = 0.0f
          EMA12 = 0.0f
          EMA26 = 0.0f
          // Momentum Indicators (will be calculated later)
          RSI = 0.0f
          MACD = 0.0f
          MACDSignal = 0.0f
          MACDHistogram = 0.0f
          // Volatility Indicators (will be calculated later)
          ATR = 0.0f
          BollingerUpper = 0.0f
          BollingerMiddle = 0.0f
          BollingerLower = 0.0f
          Volatility = 0.0f
          // Price Change Indicators
          ChangePercent = float32 item.ChangePercent
          ChangeAmount = float32 item.ChangeAmount
          // Volume Indicators (will be calculated later)
          OBV = 0.0f
          VWAP = 0.0f })
      |> Array.sortBy (fun r -> r.Date)
    | SGE rawSGEData ->
      rawSGEData
      |> Array.filter (fun item -> item.Close > 0.0)
      |> Array.map (fun item ->
        { Date = item.Date
          Open = item.Open
          High = item.High
          Low = item.Low
          Close = item.Close
          Volume = 0L // SGE data doesn't have volume
          Amount = 0.0 // SGE data doesn't have amount
          // Moving Averages (will be calculated later)
          MA3 = 0.0f
          MA5 = 0.0f
          MA9 = 0.0f
          MA20 = 0.0f
          EMA12 = 0.0f
          EMA26 = 0.0f
          // Momentum Indicators (will be calculated later)
          RSI = 0.0f
          MACD = 0.0f
          MACDSignal = 0.0f
          MACDHistogram = 0.0f
          // Volatility Indicators (will be calculated later)
          ATR = 0.0f
          BollingerUpper = 0.0f
          BollingerMiddle = 0.0f
          BollingerLower = 0.0f
          Volatility = 0.0f
          // Price Change Indicators (will be calculated later for SGE)
          ChangePercent = 0.0f
          ChangeAmount = 0.0f
          // Volume Indicators (will be calculated later)
          OBV = 0.0f
          VWAP = 0.0f })
      |> Array.sortBy (fun r -> r.Date)

  /// <summary>
  /// Validates that data arrays have compatible lengths for operations.
  /// </summary>
  /// <param name="arrays">Variable number of arrays to validate.</param>
  /// <returns>Result indicating validation success or length mismatch error.</returns>
  let validateArrayLengths (arrays : 'T[][]) =
    match arrays with
    | [||] -> Error (DataAcquisitionFailed "No arrays provided for validation")
    | arrays ->
      let firstLength = arrays.[0].Length

      let mismatched =
        arrays |> Array.exists (fun arr -> arr.Length <> firstLength)

      if mismatched then
        Error (DataAcquisitionFailed "Array lengths do not match")
      else
        Ok firstLength

  /// <summary>
  /// Processes raw gold data records by calculating comprehensive technical indicators.
  /// Aligns the data to ensure all records have valid indicator values.
  /// </summary>
  /// <param name="records">Array of gold data records with raw prices.</param>
  /// <returns>Result containing array of processed records or error.</returns>
  let processGoldData (records : GoldDataRecord[]) =
    // Define technical indicator functions locally
    let calculateEMA (values : float[]) (period : int) =
      if values.Length < period then
        Error (
          DataAcquisitionFailed
            $"Insufficient data for EMA calculation: {values.Length} values, minimum {period} required"
        )
      else
        let multiplier = 2.0 / (float period + 1.0)
        let ema = Array.zeroCreate values.Length

        // Initialize first EMA value as SMA
        let initialSMA = values.[.. period - 1] |> Array.average
        ema.[period - 1] <- initialSMA

        // Calculate subsequent EMA values
        for i in period .. values.Length - 1 do
          ema.[i] <- (values.[i] - ema.[i - 1]) * multiplier + ema.[i - 1]

        Ok ema.[period - 1 ..]

    let calculateRSI (values : float[]) (period : int) =
      if values.Length < period + 1 then
        Error (
          DataAcquisitionFailed
            $"Insufficient data for RSI calculation: {values.Length} values, minimum {period + 1} required"
        )
      else
        // Calculate price changes
        let changes = Array.zeroCreate (values.Length - 1)

        for i in 1 .. values.Length - 1 do
          changes.[i - 1] <- values.[i] - values.[i - 1]

        // Calculate gains and losses
        let gains = changes |> Array.map (fun c -> if c > 0.0 then c else 0.0)
        let losses = changes |> Array.map (fun c -> if c < 0.0 then -c else 0.0)

        let rsi = Array.zeroCreate (changes.Length - period + 1)

        for i in period - 1 .. changes.Length - 1 do
          let avgGain =
            if i = period - 1 then
              gains.[.. period - 1] |> Array.average
            else
              (rsi.[i - period] * float (period - 1) + gains.[i]) / float period

          let avgLoss =
            if i = period - 1 then
              losses.[.. period - 1] |> Array.average
            else
              (rsi.[i - period + 1] * float (period - 1) + losses.[i])
              / float period

          let rs = if avgLoss = 0.0 then 100.0 else avgGain / avgLoss
          rsi.[i - period + 1] <- 100.0 - (100.0 / (1.0 + rs))

        Ok rsi

    let calculateMACD (values : float[]) =
      if values.Length < 26 then
        Error (
          DataAcquisitionFailed
            $"Insufficient data for MACD calculation: {values.Length} values, minimum 26 required"
        )
      else
        match calculateEMA values 12, calculateEMA values 26 with
        | Ok ema12, Ok ema26 ->
          // Align EMA arrays
          let offset = ema12.Length - ema26.Length
          let alignedEMA12 = ema12.[offset..]
          let alignedEMA26 = ema26

          // Calculate MACD line
          let macd =
            Array.zip alignedEMA12 alignedEMA26
            |> Array.map (fun (e12, e26) -> e12 - e26)

          // Calculate signal line (9-period EMA of MACD)
          match calculateEMA macd 9 with
          | Ok signal ->
            let signalOffset = macd.Length - signal.Length
            let alignedMACD = macd.[signalOffset..]

            // Calculate histogram
            let histogram =
              Array.zip alignedMACD signal |> Array.map (fun (m, s) -> m - s)

            Ok (alignedMACD, signal, histogram)
          | Error err -> Error err
        | Error err, _ -> Error err
        | _, Error err -> Error err

    let calculateATR
      (highs : float[])
      (lows : float[])
      (closes : float[])
      (period : int)
      =
      match
        validateArrayLengths
          [| highs
             lows
             closes |]
      with
      | Error err -> Error err
      | Ok _ ->
        if closes.Length < period + 1 then
          Error (
            DataAcquisitionFailed
              $"Insufficient data for ATR calculation: {closes.Length} values, minimum {period + 1} required"
          )
        else
          // Calculate True Range for each period
          let tr = Array.zeroCreate (closes.Length - 1)

          for i in 1 .. closes.Length - 1 do
            let tr1 = highs.[i] - lows.[i]
            let tr2 = abs (highs.[i] - closes.[i - 1])
            let tr3 = abs (lows.[i] - closes.[i - 1])
            tr.[i - 1] <- max tr1 (max tr2 tr3)

          // Calculate ATR using exponential moving average
          let atr = Array.zeroCreate (tr.Length - period + 1)

          // Initialize first ATR value
          atr.[0] <- tr.[.. period - 1] |> Array.average

          // Calculate subsequent ATR values
          for i in period .. tr.Length - 1 do
            atr.[i - period + 1] <-
              (atr.[i - period] * float (period - 1) + tr.[i]) / float period

          Ok atr

    let calculateBollingerBands
      (values : float[])
      (period : int)
      (stdDev : float)
      =
      if values.Length < period then
        Error (
          DataAcquisitionFailed
            $"Insufficient data for Bollinger Bands calculation: {values.Length} values, minimum {period} required"
        )
      else
        let bands = Array.zeroCreate (values.Length - period + 1)

        for i in 0 .. values.Length - period do
          let window = values.[i .. i + period - 1]
          let middle = Array.average window
          let std = Statistics.StandardDeviation window
          let upper = middle + stdDev * std
          let lower = middle - stdDev * std

          bands.[i] <- (upper, middle, lower)

        let upperBand = bands |> Array.map (fun (u, _, _) -> u)
        let middleBand = bands |> Array.map (fun (_, m, _) -> m)
        let lowerBand = bands |> Array.map (fun (_, _, l) -> l)

        Ok (upperBand, middleBand, lowerBand)

    let calculateHistoricalVolatility (values : float[]) (period : int) =
      if values.Length < period + 1 then
        Error (
          DataAcquisitionFailed
            $"Insufficient data for volatility calculation: {values.Length} values, minimum {period + 1} required"
        )
      else
        // Calculate logarithmic returns
        let returns = Array.zeroCreate (values.Length - 1)

        for i in 1 .. values.Length - 1 do
          returns.[i - 1] <- log (values.[i] / values.[i - 1])

        // Calculate rolling volatility
        let volatility = Array.zeroCreate (returns.Length - period + 1)

        for i in 0 .. returns.Length - period do
          let window = returns.[i .. i + period - 1]
          volatility.[i] <- Statistics.StandardDeviation window

        Ok volatility

    let calculateOBV (closes : float[]) (volumes : int64[]) =
      match validateArrayLengths [| closes ; volumes |> Array.map float |] with
      | Error err -> Error err
      | Ok _ ->
        let obv = Array.zeroCreate closes.Length
        obv.[0] <- float volumes.[0]

        for i in 1 .. closes.Length - 1 do
          if closes.[i] > closes.[i - 1] then
            obv.[i] <- obv.[i - 1] + float volumes.[i]
          elif closes.[i] < closes.[i - 1] then
            obv.[i] <- obv.[i - 1] - float volumes.[i]
          else
            obv.[i] <- obv.[i - 1]

        Ok obv

    let calculateVWAP
      (highs : float[])
      (lows : float[])
      (closes : float[])
      (volumes : int64[])
      =
      match
        validateArrayLengths
          [| highs
             lows
             closes
             volumes |> Array.map float |]
      with
      | Error err -> Error err
      | Ok _ ->
        let vwap = Array.zeroCreate highs.Length

        for i in 0 .. highs.Length - 1 do
          let typicalPrice = (highs.[i] + lows.[i] + closes.[i]) / 3.0

          if i = 0 then
            vwap.[i] <- typicalPrice
          else
            let cumulativeTPV =
              vwap.[i - 1] * float (i) + typicalPrice * float volumes.[i]

            let cumulativeVolume = float (i + 1) * float volumes.[i] // Simplified calculation
            vwap.[i] <- cumulativeTPV / cumulativeVolume

        Ok vwap

    if records.Length < 50 then // Need sufficient data for all indicators
      Error (
        DataAcquisitionFailed
          $"Insufficient data for technical indicators: {records.Length} records, minimum 50 required"
      )
    else
      // Extract price arrays
      let closeValues = records |> Array.map (fun r -> r.Close)
      let highValues = records |> Array.map (fun r -> r.High)
      let lowValues = records |> Array.map (fun r -> r.Low)
      let openValues = records |> Array.map (fun r -> r.Open)
      let volumeValues = records |> Array.map (fun r -> r.Volume)

      // Calculate only the indicators we need
      let ma3Result = calculateMovingAverage closeValues 3
      let ma9Result = calculateMovingAverage closeValues 9
      let ma20Result = calculateMovingAverage closeValues 20
      let rsiResult = calculateRSI closeValues 14
      let atrResult = calculateATR highValues lowValues closeValues 14
      let volResult = calculateHistoricalVolatility closeValues 14

      // Check if all calculations succeeded
      let allResults =
        [ ma3Result
          ma9Result
          ma20Result
          rsiResult
          atrResult
          volResult ]

      let hasErrors =
        allResults
        |> List.exists (function
          | Error _ -> true
          | _ -> false)

      if hasErrors then
        Error (DataAcquisitionFailed "Failed to calculate technical indicators")
      else
        // Extract successful results
        let ma3 =
          match ma3Result with
          | Ok arr -> arr
          | _ -> [||]

        let ma9 =
          match ma9Result with
          | Ok arr -> arr
          | _ -> [||]

        let ma20 =
          match ma20Result with
          | Ok arr -> arr
          | _ -> [||]

        let rsi =
          match rsiResult with
          | Ok arr -> arr
          | _ -> [||]

        let atr =
          match atrResult with
          | Ok arr -> arr
          | _ -> [||]

        let volatility =
          match volResult with
          | Ok arr -> arr
          | _ -> [||]

        // Find the maximum offset to align all indicators
        let offsets =
          [ closeValues.Length - ma3.Length
            closeValues.Length - ma9.Length
            closeValues.Length - ma20.Length
            closeValues.Length - rsi.Length
            closeValues.Length - atr.Length
            closeValues.Length - volatility.Length ]

        let maxOffset = if offsets.Length > 0 then offsets |> List.max else 0
        let alignedLength = closeValues.Length - maxOffset

        Ok (
          Array.init alignedLength (fun i ->
            let dataIndex = i + maxOffset

            { records.[dataIndex] with
                // Moving Averages
                MA3 =
                  if dataIndex >= closeValues.Length - ma3.Length then
                    float32 ma3.[dataIndex - (closeValues.Length - ma3.Length)]
                  else
                    0.0f
                MA5 = 0.0f // Not used
                MA9 =
                  if dataIndex >= closeValues.Length - ma9.Length then
                    float32 ma9.[dataIndex - (closeValues.Length - ma9.Length)]
                  else
                    0.0f
                MA20 =
                  if dataIndex >= closeValues.Length - ma20.Length then
                    float32
                      ma20.[dataIndex - (closeValues.Length - ma20.Length)]
                  else
                    0.0f
                EMA12 = 0.0f // Not used
                EMA26 = 0.0f // Not used
                // Momentum Indicators
                RSI =
                  if dataIndex >= closeValues.Length - rsi.Length then
                    float32 rsi.[dataIndex - (closeValues.Length - rsi.Length)]
                  else
                    0.0f
                MACD = 0.0f // Not used
                MACDSignal = 0.0f // Not used
                MACDHistogram = 0.0f // Not used
                // Volatility Indicators
                ATR =
                  if dataIndex >= closeValues.Length - atr.Length then
                    float32 atr.[dataIndex - (closeValues.Length - atr.Length)]
                  else
                    0.0f
                BollingerUpper = 0.0f // Not used
                BollingerMiddle = 0.0f // Not used
                BollingerLower = 0.0f // Not used
                Volatility =
                  if dataIndex >= closeValues.Length - volatility.Length then
                    float32
                      volatility.[dataIndex
                                  - (closeValues.Length - volatility.Length)]
                  else
                    0.0f
                // Volume Indicators (set to 0 for now)
                OBV = 0.0f
                VWAP = 0.0f })
        )

  /// <summary>
  /// Splits data into training and testing sets based on the specified ratio.
  /// </summary>
  /// <param name="records">Array of data records to split.</param>
  /// <param name="trainRatio">Ratio of data to use for training (0.0 to 1.0).</param>
  /// <returns>Tuple of (training data, testing data).</returns>
  let splitData (records : 'T[]) (trainRatio : float) =
    let trainSize = int (float records.Length * trainRatio)
    let trainData = records.[.. trainSize - 1]
    let testData = records.[trainSize..]
    trainData, testData

  /// <summary>
  /// <summary>
  /// Performs comprehensive data quality checks on gold price data.
  /// </summary>
  /// <param name="records">Array of gold data records to validate.</param>
  /// <returns>Result containing validated records or error with details.</returns>
  let validateDataQuality (records : GoldDataRecord[]) =
    if records.Length = 0 then
      Error (DataAcquisitionFailed "No data records provided")
    else
      let prices = records |> Array.map (fun r -> r.Close)

      // Check for invalid prices
      let invalidPrices =
        prices
        |> Array.filter (fun p ->
          p <= 0.0 || System.Double.IsNaN p || System.Double.IsInfinity p)

      if invalidPrices.Length > 0 then
        Error (
          DataAcquisitionFailed
            $"Found {invalidPrices.Length} invalid price values"
        )
      else
        // Check for price outliers using IQR method
        let sortedPrices = prices |> Array.sort
        let q1 = sortedPrices.[int (float prices.Length / 4.0)]
        let q3 = sortedPrices.[int (float prices.Length * 3.0 / 4.0)]
        let iqr = q3 - q1
        let lowerBound = q1 - 1.5 * iqr
        let upperBound = q3 + 1.5 * iqr

        let outliers =
          prices |> Array.filter (fun p -> p < lowerBound || p > upperBound)

        if outliers.Length > int (float prices.Length * 0.1) then // More than 10% outliers
          Error (
            DataAcquisitionFailed
              $"Too many price outliers detected: {outliers.Length} out of {prices.Length}"
          )
        else
          // Check for chronological order
          let datesInOrder =
            records
            |> Array.pairwise
            |> Array.forall (fun (a, b) -> a.Date <= b.Date)

          if not datesInOrder then
            Error (
              DataAcquisitionFailed
                "Data records are not in chronological order"
            )
          else
            // Check for duplicate dates
            let uniqueDates =
              records |> Array.map (fun r -> r.Date) |> Array.distinct

            if uniqueDates.Length < records.Length then
              Error (DataAcquisitionFailed "Duplicate dates found in data")
            else
              Ok records

  /// <summary>
  /// Detects and removes data anomalies using statistical methods.
  /// </summary>
  /// <param name="records">Array of gold data records.</param>
  /// <returns>Result containing cleaned records or error.</returns>
  let removeAnomalies (records : GoldDataRecord[]) =
    if records.Length < 10 then
      Ok records // Not enough data for anomaly detection
    else
      let prices = records |> Array.map (fun r -> r.Close)

      // Simple moving median filter for smoothing
      let windowSize = min 5 (prices.Length / 3)

      let smoothedPrices =
        Array.init prices.Length (fun i ->
          let start = max 0 (i - windowSize / 2)
          let endIdx = min (prices.Length - 1) (i + windowSize / 2)
          let window = prices.[start..endIdx]
          let sorted = Array.sort window
          sorted.[int (float window.Length / 2.0)] // Median
        )

      // Calculate residuals
      let residuals =
        Array.zip prices smoothedPrices
        |> Array.map (fun (actual, smoothed) -> abs (actual - smoothed))

      // Remove points with residuals > 3 * median absolute deviation
      let medianResidual =
        Array.sort residuals |> fun arr -> arr.[arr.Length / 2]

      let threshold = 3.0 * medianResidual

      let filteredRecords =
        Array.zip records residuals
        |> Array.filter (fun (_, residual) -> residual <= threshold)
        |> Array.map fst

      if filteredRecords.Length < int (float records.Length * 0.8) then // Removed more than 20%
        Error (
          DataAcquisitionFailed
            $"Too many anomalies detected, removed {records.Length - filteredRecords.Length} records"
        )
      else
        logInfo
          $"Removed {records.Length - filteredRecords.Length} anomalous data points"

        Ok filteredRecords

/// <summary>
/// <summary>
/// <summary>
/// <summary>
