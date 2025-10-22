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
          Close = item.Close
          MA3 = 0.0f // Will be calculated later
          MA9 = 0.0f }) // Will be calculated later
      |> Array.sortBy (fun r -> r.Date)
    | SGE rawSGEData ->
      rawSGEData
      |> Array.filter (fun item -> item.Close > 0.0)
      |> Array.map (fun item ->
        { Date = item.Date
          Close = item.Close // Use close price for consistency
          MA3 = 0.0f // Will be calculated later
          MA9 = 0.0f }) // Will be calculated later
      |> Array.sortBy (fun r -> r.Date)

  /// <summary>
  /// Processes raw gold data records by calculating moving averages.
  /// Aligns the data to ensure all records have valid moving average values.
  /// </summary>
  /// <param name="records">Array of gold data records with raw prices.</param>
  /// <returns>Result containing array of processed records or error.</returns>
  let processGoldData (records : GoldDataRecord[]) =
    if records.Length < 9 then
      Error (
        DataAcquisitionFailed
          $"Insufficient data for moving averages: {records.Length} records, minimum 9 required"
      )
    else
      let closeValues = records |> Array.map (fun r -> r.Close)

      match
        calculateMovingAverage closeValues 3,
        calculateMovingAverage closeValues 9
      with
      | Ok ma3Values, Ok ma9Values ->
        let ma3Float32 = ma3Values |> Array.map float32
        let ma9Float32 = ma9Values |> Array.map float32

        // Align data: moving averages reduce the number of data points
        let offset3 = closeValues.Length - ma3Float32.Length
        let offset9 = closeValues.Length - ma9Float32.Length
        let maxOffset = max offset3 offset9
        let alignedLength = closeValues.Length - maxOffset

        Ok (
          Array.init alignedLength (fun i ->
            let dataIndex = i + maxOffset

            { records.[dataIndex] with
                MA3 =
                  if dataIndex >= offset3 then
                    ma3Float32.[dataIndex - offset3]
                  else
                    0.0f
                MA9 =
                  if dataIndex >= offset9 then
                    ma9Float32.[dataIndex - offset9]
                  else
                    0.0f })
        )
      | Error err, _ -> Error err
      | _, Error err -> Error err

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
  /// Performs data imputation for missing values using interpolation.
  /// </summary>
  /// <param name="records">Array of gold data records.</param>
  /// <returns>Result containing imputed records or error.</returns>
  let imputeMissingValues (records : GoldDataRecord[]) =
    // For now, just validate that we don't have missing values
    // In a real system, you might interpolate missing prices
    let hasMissing = records |> Array.exists (fun r -> r.Close = 0.0)

    if hasMissing then
      Error (DataAcquisitionFailed "Missing values detected in price data")
    else
      Ok records
