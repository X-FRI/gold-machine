namespace GoldMachine

open Plotly.NET

/// <summary>
/// Visualization module for creating charts and plots of price data and strategy performance.
/// Provides functions to generate interactive HTML charts using Plotly.NET.
/// </summary>
module Visualization =

  /// <summary>
  /// Creates a line chart for price data over time.
  /// </summary>
  /// <param name="dates">Array of dates for the x-axis.</param>
  /// <param name="prices">Array of price values for the y-axis.</param>
  /// <param name="title">Chart title.</param>
  /// <returns>A Plotly chart object.</returns>
  let plotPrices
    (dates : System.DateTime[])
    (prices : float[])
    (title : string)
    =
    Chart.Line (dates, prices, Name = title)

  /// <summary>
  /// Creates a line chart for cumulative returns over time.
  /// </summary>
  /// <param name="dates">Array of dates for the x-axis.</param>
  /// <param name="returns">Array of cumulative return values.</param>
  /// <param name="title">Chart title.</param>
  /// <returns>A Plotly chart object.</returns>
  let plotCumulativeReturns
    (dates : System.DateTime[])
    (returns : float[])
    (title : string)
    =
    Chart.Line (dates, returns, Name = title)

  /// <summary>
  /// Creates a comparison chart showing both actual and predicted prices.
  /// </summary>
  /// <param name="dates">Array of dates for both series.</param>
  /// <param name="actualPrices">Array of actual price values.</param>
  /// <param name="predictedPrices">Array of predicted price values.</param>
  /// <returns>A combined Plotly chart object.</returns>
  let plotPriceComparison
    (dates : System.DateTime[])
    (actualPrices : float[])
    (predictedPrices : float[])
    =
    let actualChart = Chart.Line (dates, actualPrices, Name = "Actual Price")

    let predictedChart =
      Chart.Line (dates, predictedPrices, Name = "Predicted Price")

    Chart.combine [ actualChart ; predictedChart ]

  /// <summary>
  /// Creates a chart showing trading signals over time.
  /// </summary>
  /// <param name="dates">Array of dates.</param>
  /// <param name="signals">Array of trading signals (1.0 for buy, 0.0 for hold).</param>
  /// <returns>A Plotly chart object.</returns>
  let plotTradingSignals
    (dates : System.DateTime[])
    (signals : float[])
    (title : string)
    =
    Chart.Line (dates, signals, Name = title)

  /// <summary>
  /// Saves a chart to an HTML file.
  /// </summary>
  /// <param name="chart">The chart to save.</param>
  /// <param name="filePath">Path to the output HTML file.</param>
  /// <returns>Result indicating success or file operation error.</returns>
  let saveChart (chart : GenericChart) (filePath : string) =
    try
      Ok (Chart.saveHtml filePath chart)
    with ex ->
      Error (
        FileOperationFailed $"Failed to save chart to {filePath}: {ex.Message}"
      )

  /// <summary>
  /// Creates and saves all standard analysis charts.
  /// </summary>
  /// <param name="testDates">Dates for the test period.</param>
  /// <param name="actualPrices">Actual price data.</param>
  /// <param name="predictedPrices">Predicted price data.</param>
  /// <param name="cumulativeReturns">Strategy cumulative returns.</param>
  /// <returns>Result indicating success or error in chart generation.</returns>
  let generateAnalysisCharts
    (testDates : System.DateTime[])
    (actualPrices : float32[])
    (predictedPrices : float32[])
    (cumulativeReturns : float[])
    =
    try
      let actualPricesFloat = actualPrices |> Array.map float
      let predictedPricesFloat = predictedPrices |> Array.map float

      let priceComparisonChart =
        plotPriceComparison testDates actualPricesFloat predictedPricesFloat

      let returnsChart =
        plotCumulativeReturns
          testDates
          cumulativeReturns
          "Strategy Cumulative Returns"

      match saveChart priceComparisonChart "gold_price_prediction.html" with
      | Error err -> Error err
      | Ok _ ->
        match saveChart returnsChart "cumulative_returns.html" with
        | Error err -> Error err
        | Ok _ -> Ok ()
    with ex ->
      Error (FileOperationFailed $"Chart generation failed: {ex.Message}")

  /// <summary>
  /// Creates a summary chart showing multiple performance metrics.
  /// </summary>
  /// <param name="dates">Array of dates.</param>
  /// <param name="prices">Price data.</param>
  /// <param name="returns">Return data.</param>
  /// <param name="signals">Trading signals.</param>
  /// <returns>A combined multi-series chart.</returns>
  let createPerformanceDashboard
    (dates : System.DateTime[])
    (prices : float[])
    (returns : float[])
    (signals : float[])
    =
    let priceChart = plotPrices dates prices "GLD Price"
    let returnsChart = plotCumulativeReturns dates returns "Cumulative Returns"
    let signalChart = plotTradingSignals dates signals "Trading Signals"

    Chart.combine
      [ priceChart
        returnsChart
        signalChart ]
    |> Chart.withTitle "Gold Price Prediction Performance Dashboard"

  /// <summary>
  /// Validates chart input data for consistency.
  /// </summary>
  /// <param name="dates">Date array.</param>
  /// <param name="values">Value arrays to validate against dates.</param>
  /// <returns>Result indicating validation success or data inconsistency error.</returns>
  let validateChartData (dates : System.DateTime[]) (values : float[][]) =
    let lengths =
      Array.concat [| [| dates.Length |] ; values |> Array.map Array.length |]

    match DataProcessing.validateArrayLengths [| lengths |] with
    | Error _ ->
      Error (ConfigurationError "Chart data arrays have inconsistent lengths")
    | Ok _ -> Ok ()
