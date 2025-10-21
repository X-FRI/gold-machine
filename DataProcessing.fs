namespace GoldMachine

open MathNet.Numerics.Statistics

/// <summary>
/// Data processing module for calculating technical indicators and statistical measures.
/// Provides functions for moving averages, returns calculations, and data transformations.
/// </summary>
module DataProcessing =

  /// <summary>
  /// Calculates simple moving average for the given window size.
  /// </summary>
  /// <param name="values">Array of price values.</param>
  /// <param name="windowSize">Size of the moving average window.</param>
  /// <returns>Array of moving average values.</returns>
  let calculateMovingAverage (values : float[]) (windowSize : int) =
    if values.Length < windowSize then
      [||]
    else
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
  /// Processes raw gold data records by calculating moving averages.
  /// Aligns the data to ensure all records have valid moving average values.
  /// </summary>
  /// <param name="records">Array of gold data records with raw prices.</param>
  /// <returns>Array of processed records with calculated moving averages.</returns>
  let processGoldData (records : GoldDataRecord[]) =
    let closeValues = records |> Array.map (fun r -> r.Close)
    let ma3Values = calculateMovingAverage closeValues 3 |> Array.map float32
    let ma9Values = calculateMovingAverage closeValues 9 |> Array.map float32

    // Align data: moving averages reduce the number of data points
    let offset3 = closeValues.Length - ma3Values.Length
    let offset9 = closeValues.Length - ma9Values.Length
    let maxOffset = max offset3 offset9
    let alignedLength = closeValues.Length - maxOffset

    Array.init alignedLength (fun i ->
      let dataIndex = i + maxOffset

      { records.[dataIndex] with
          MA3 =
            if dataIndex >= offset3 then
              ma3Values.[dataIndex - offset3]
            else
              0.0f
          MA9 =
            if dataIndex >= offset9 then
              ma9Values.[dataIndex - offset9]
            else
              0.0f })

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
