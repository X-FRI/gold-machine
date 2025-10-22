namespace GoldMachine

open System

/// <summary>
/// Main program module for the Gold Price Prediction System.
/// Orchestrates data acquisition, model training, evaluation, and trading strategy execution.
/// </summary>
module Program =

  /// <summary>
  /// Logs information messages to the console with timestamp.
  /// </summary>
  /// <param name="message">The message to log.</param>
  let logInfo (message : string) =
    printfn
      "[%s] INFO: %s"
      (DateTime.Now.ToString ("yyyy-MM-dd HH:mm:ss"))
      message

  /// <summary>
  /// Logs error messages to the console with timestamp.
  /// </summary>
  /// <param name="message">The error message to log.</param>
  let logError (message : string) =
    printfn
      "[%s] ERROR: %s"
      (DateTime.Now.ToString ("yyyy-MM-dd HH:mm:ss"))
      message

  /// <summary>
  /// Handles and logs errors that occur during program execution.
  /// </summary>
  /// <param name="error">The error to handle.</param>
  let handleError (error : GoldMachineError) =
    match error with
    | InvalidDateRange msg -> logError (sprintf "Invalid date range: %s" msg)
    | DataAcquisitionFailed msg ->
      logError (sprintf "Data acquisition failed: %s" msg)
    | ModelTrainingFailed msg ->
      logError (sprintf "Model training failed: %s" msg)
    | FileOperationFailed msg ->
      logError (sprintf "File operation failed: %s" msg)
    | ConfigurationError msg -> logError (sprintf "Configuration error: %s" msg)

  /// <summary>
  /// Runs the complete gold price prediction workflow.
  /// </summary>
  /// <param name="config">Configuration for the prediction system.</param>
  /// <returns>Result indicating success or failure of the workflow.</returns>
  let runPredictionWorkflow (config : GoldMachineConfig) =
    async {
      try
        logInfo "Starting Gold Price Prediction System..."

        // Validate configuration
        match Configuration.validateConfig config with
        | Error err -> return Error err
        | Ok validatedConfig ->

          // Acquire data from configured provider
          let provider =
            DataProviders.DataProviderFactory.getProviderFromConfig
              validatedConfig

          logInfo $"Acquiring gold price data from {provider.Name}..."

          match! DataAcquisition.acquireGoldData provider validatedConfig with
          | Error err -> return Error err
          | Ok rawRecords ->

            // Validate data sufficiency
            match DataAcquisition.validateData rawRecords 50 with
            | Error err -> return Error err
            | Ok validatedRecords ->

              // Process data (calculate moving averages)
              logInfo "Processing data and calculating technical indicators..."

              let processedRecords =
                DataProcessing.processGoldData validatedRecords

              logInfo (
                sprintf
                  "Data processed successfully. Records: %d"
                  processedRecords.Length
              )

              // Split data into training and testing sets
              logInfo (
                sprintf
                  "Splitting data (%.1f%% training, %.1f%% testing)..."
                  (config.TrainRatio * 100.0)
                  ((1.0 - config.TrainRatio) * 100.0)
              )

              let trainData, testData =
                DataProcessing.splitData processedRecords config.TrainRatio

              logInfo (
                sprintf
                  "Training set: %d records, Test set: %d records"
                  trainData.Length
                  testData.Length
              )

              // Train the machine learning model
              logInfo "Training linear regression model..."
              let mlContext = MachineLearning.createMLContext ()

              let model =
                MachineLearning.trainLinearRegression mlContext trainData

              match MachineLearning.validateModel model with
              | Error err -> return Error err
              | Ok validatedModel ->

                logInfo "Model trained successfully"

                // Evaluate model performance
                logInfo "Evaluating model performance..."

                let testInputs =
                  testData
                  |> Array.map MachineLearning.createPredictionInput
                  |> Array.toSeq

                let actualPrices =
                  testData |> Array.map (fun r -> float32 r.Close)

                let evaluation =
                  MachineLearning.evaluateModel
                    validatedModel
                    testInputs
                    actualPrices

                logInfo (sprintf "Model R² Score: %.4f" evaluation.RSquared)
                logInfo (sprintf "Model MAE: %.4f" evaluation.MAE)
                logInfo (sprintf "Model RMSE: %.4f" evaluation.RMSE)
                logInfo (sprintf "Model MAPE: %.2f%%" evaluation.MAPE)

                // Model health assessment
                logInfo "Performing model health assessment..."
                let healthReport = MachineLearning.assessModelHealth evaluation
                logInfo (sprintf "Model Health Status: %A" healthReport.Status)
                logInfo (sprintf "Health Message: %s" healthReport.Message)
                logInfo (sprintf "Risk Level: %.2f" healthReport.RiskLevel)

                if healthReport.Recommendations.Length > 0 then
                  logInfo "Recommendations:"

                  healthReport.Recommendations
                  |> List.iter (fun recommendation ->
                    logInfo (sprintf "  - %s" recommendation))

                // Generate predictions for test data
                logInfo "Generating price predictions..."

                let predictions =
                  MachineLearning.predictBatch validatedModel testInputs

                // Evaluate trading strategy
                logInfo "Evaluating advanced trading strategy..."

                let (signals, strategyReturns, cumulativeReturns, sharpeRatio) =
                  TradingStrategy.evaluateStrategy
                    predictions
                    actualPrices
                    config

                logInfo (sprintf "Strategy Sharpe Ratio: %.4f" sharpeRatio)

                // Advanced signal generation and market environment analysis
                logInfo "Generating advanced trading signals..."

                let (weightedSignals, positionSizes, marketRegime) =
                  TradingStrategy.generateAdvancedSignals
                    predictions
                    actualPrices
                    evaluation

                logInfo (sprintf "Market Regime: %A" marketRegime)

                // Position sizing demonstration
                logInfo "Calculating position sizing with Kelly Criterion..."
                let expectedReturn = Array.average strategyReturns

                let positionSizing =
                  TradingStrategy.calculatePositionSizingMAE
                    expectedReturn
                    config.RiskFreeRate
                    evaluation.MAE
                    (Array.average (actualPrices |> Array.map float))

                logInfo (
                  sprintf
                    "Kelly Position Size: %.2f%%"
                    (positionSizing.PositionSize * 100.0)
                )

                logInfo (
                  sprintf
                    "Estimated Max Drawdown: %.2f%%"
                    (positionSizing.MaxDrawdown * 100.0)
                )

                // Get latest data for prediction interval estimation
                let latestTestData = testData.[testData.Length - 1]

                // Prediction interval estimation
                logInfo "Estimating prediction intervals..."
                let latestPrediction = predictions.[predictions.Length - 1]

                let predictionInterval =
                  MachineLearning.estimatePredictionInterval
                    latestPrediction
                    evaluation.RMSE
                    0.95

                let adjustedInterval =
                  MachineLearning.adjustIntervalForPriceRange
                    predictionInterval
                    evaluation.MAPE
                    latestTestData.Close

                logInfo (
                  sprintf
                    "95%% Prediction Interval: [%.2f, %.2f]"
                    (fst predictionInterval)
                    (snd predictionInterval)
                )

                logInfo (
                  sprintf
                    "MAPE-adjusted Interval: [%.2f, %.2f]"
                    (fst adjustedInterval)
                    (snd adjustedInterval)
                )

                // Data quality monitoring
                logInfo "Monitoring data quality..."

                let recentPrices =
                  testData
                  |> Array.map (fun r -> r.Close)
                  |> Array.take (min 20 testData.Length)

                let dataQuality =
                  TradingStrategy.monitorDataQuality
                    recentPrices
                    evaluation.MAPE

                logInfo (sprintf "Data Quality: %s" dataQuality.Message)

                // Generate visualization charts
                logInfo "Generating analysis charts..."
                let testDates = testData |> Array.map (fun r -> r.Date)

                match
                  Visualization.generateAnalysisCharts
                    testDates
                    actualPrices
                    predictions
                    cumulativeReturns
                with
                | Error err -> return Error err
                | Ok _ ->

                  logInfo "Charts saved successfully"

                  // Generate trading recommendation for latest data
                  logInfo "Generating trading recommendation..."

                  let latestData =
                    processedRecords.[processedRecords.Length - 1]

                  let predictedPrice =
                    MachineLearning.predict
                      validatedModel
                      (MachineLearning.createPredictionInput latestData)

                  let recommendation =
                    TradingStrategy.generateTradingRecommendation
                      latestData.Close
                      predictedPrice

                  logInfo (
                    sprintf
                      "Latest data date: %s"
                      (latestData.Date.ToString ("yyyy-MM-dd"))
                  )

                  logInfo (sprintf "Current price: %.2f" latestData.Close)
                  logInfo (sprintf "Predicted next price: %.2f" predictedPrice)
                  logInfo (sprintf "Trading recommendation: %s" recommendation)

                  logInfo "Gold Price Prediction System completed successfully"
                  return Ok ()
      with ex ->
        return
          Error (
            DataAcquisitionFailed (
              sprintf "Workflow execution failed: %s" ex.Message
            )
          )
    }

  /// <summary>
  /// Parses command line arguments and creates appropriate configuration.
  /// </summary>
  /// <param name="argv">Command line arguments.</param>
  /// <returns>GoldMachineConfig for the specified data provider.</returns>
  let createConfigFromArgs (argv : string[]) =
    let mutable etfSymbol = "518880" // Default ETF symbol (GLD ETF)
    let mutable useSGE = false

    // Parse command line arguments
    let rec parseArgs index =
      if index < argv.Length then
        match argv.[index].ToLower () with
        | "--etf" ->
          if index + 1 < argv.Length then
            etfSymbol <- argv.[index + 1]
            parseArgs (index + 2)
          else
            logError "Error: --etf requires a symbol argument"
            parseArgs (index + 1)
        | "sge" ->
          useSGE <- true
          parseArgs (index + 1)
        | arg when arg.StartsWith ("--") ->
          logError $"Warning: Unknown option '{arg}'"
          parseArgs (index + 1)
        | arg ->
          logError $"Warning: Unknown argument '{arg}'"
          parseArgs (index + 1)
      else
        ()

    parseArgs 0

    // Create configuration based on parsed arguments
    if useSGE then
      logInfo "Using Shanghai Gold Exchange data provider"
      Configuration.getSGEConfig ()
    else
      if etfSymbol <> "518880" then
        logInfo $"Using custom ETF symbol: {etfSymbol}"
      else
        logInfo "Using default ETF data provider (GLD ETF)"

      Configuration.getETFConfig etfSymbol

  /// <summary>
  /// Main entry point for the Gold Price Prediction System.
  /// </summary>
  /// <param name="argv">Command line arguments for data source configuration.</param>
  /// <returns>Exit code (0 for success, 1 for failure).</returns>
  [<EntryPoint>]
  let main argv =
    printfn "Gold Price Prediction System v2.0"
    printfn "==================================="
    printfn "Usage: dotnet run [options]"
    printfn ""
    printfn "Options:"

    printfn
      "  --etf <symbol>    Use ETF data with custom symbol (default: 518880 for GLD ETF)"

    printfn "  sge               Use Shanghai Gold Exchange data"
    printfn "  (no args)         Use default ETF data (518880)"
    printfn ""
    printfn "Examples:"
    printfn "  dotnet run                    # Use default GLD ETF (518880)"
    printfn "  dotnet run --etf 159941       # Use custom ETF symbol"
    printfn "  dotnet run sge                # Use Shanghai Gold Exchange data"
    printfn ""

    let config = createConfigFromArgs argv

    match Async.RunSynchronously (runPredictionWorkflow config) with
    | Ok _ ->
      printfn "Done."
      0
    | Error err ->
      handleError err
      printfn "Failed!"
      1
