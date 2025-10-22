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

              let qualityValidatedRecords = validatedRecords // Skip quality validation for now
              let cleanedRecords = qualityValidatedRecords // Skip anomaly removal for now
              let imputedRecords = cleanedRecords // Skip imputation for now

              match DataProcessing.processGoldData imputedRecords with
              | Error err -> return Error err
              | Ok processedRecords ->

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
                let mlContext = MachineLearning.createMLContext ()

                if config.UseEnsemble then
                  logInfo
                    "Training ensemble model with all available algorithms..."

                  match MachineLearning.trainEnsembleModel trainData config with
                  | Error err -> return Error err
                  | Ok ensembleModel ->

                    match
                      MachineLearning.validateEnsembleModel ensembleModel
                    with
                    | Error err -> return Error err
                    | Ok validatedEnsemble ->

                      logInfo "Ensemble model trained successfully"

                      // Evaluate ensemble model performance
                      logInfo "Evaluating ensemble model performance..."

                      let testInputs =
                        testData
                        |> Array.map MachineLearning.createPredictionInput
                        |> Array.toSeq

                      let actualPrices =
                        testData |> Array.map (fun r -> float32 r.Close)

                      let ensembleEvaluation =
                        MachineLearning.evaluateEnsembleModel
                          validatedEnsemble
                          testInputs
                          actualPrices

                      logInfo
                        $"Ensemble R² Score: {ensembleEvaluation.EnsembleRSquared:F4}"

                      logInfo
                        $"Ensemble MAE: {ensembleEvaluation.EnsembleMAE:F4}"

                      logInfo
                        $"Ensemble RMSE: {ensembleEvaluation.EnsembleRMSE:F4}"

                      logInfo
                        $"Ensemble MAPE: {ensembleEvaluation.EnsembleMAPE:P2}"

                      // Log individual model performances
                      logInfo "Individual model performances:"

                      ensembleEvaluation.IndividualEvaluations
                      |> List.iter (fun (alg, eval) ->
                        logInfo
                          $"  {alg}: R²={eval.RSquared:F4}, MAE={eval.MAE:F4}, RMSE={eval.RMSE:F4}, MAPE={eval.MAPE:P2}")

                      // Continue with trading strategy using ensemble model
                      let currentRecord =
                        processedRecords.[processedRecords.Length - 1]

                      let currentInput =
                        MachineLearning.createPredictionInput currentRecord

                      let predictedPrice =
                        MachineLearning.predictWithEnsemble
                          validatedEnsemble
                          currentInput

                      logInfo
                        $"Current price: {currentRecord.Close:F4}, Predicted price: {predictedPrice:F4}"

                      let signals =
                        TradingStrategy.generateSimpleSignals
                          [| predictedPrice |]
                          [| float32 currentRecord.Close |]

                      let signal =
                        if signals.Length > 0 then float signals.[0] else 0.0

                      let recommendation =
                        TradingStrategy.generateTradingRecommendation
                          currentRecord.Close
                          predictedPrice

                      logInfo $"{recommendation}"

                      // Generate visualizations
                      logInfo "Generating price prediction visualization..."

                      let allInputs =
                        processedRecords
                        |> Array.map MachineLearning.createPredictionInput
                        |> Array.toSeq

                      let allPredictions =
                        MachineLearning.predictBatchWithEnsemble
                          validatedEnsemble
                          allInputs

                      let actualPricesAll =
                        processedRecords |> Array.map (fun r -> float r.Close)

                      match
                        Visualization.generateAnalysisCharts
                          (processedRecords |> Array.map (fun r -> r.Date))
                          (processedRecords
                           |> Array.map (fun r -> float32 r.Close))
                          allPredictions
                          [||] // Empty cumulative returns for now
                      with
                      | Ok _ -> logInfo "Charts generated successfully"
                      | Error err -> logInfo $"Chart generation failed: {err}"

                      // Calculate and display trading strategy metrics
                      let returns =
                        DataProcessing.calculatePercentageChange actualPricesAll

                      let strategyReturns =
                        TradingStrategy.calculateStrategyReturns
                          returns
                          [| signal |]

                      let cumulativeReturns =
                        DataProcessing.calculateCumulativeReturns
                          strategyReturns

                      let sharpeRatio =
                        DataProcessing.calculateSharpeRatio
                          strategyReturns
                          config.RiskFreeRate

                      let totalReturn =
                        if cumulativeReturns.Length > 0 then
                          cumulativeReturns.[cumulativeReturns.Length - 1]
                        else
                          0.0

                      let annualizedReturn =
                        totalReturn / float processedRecords.Length * 252.0 // Assuming 252 trading days per year

                      let maxDrawdown = 0.0 // Placeholder

                      logInfo $"Strategy Sharpe Ratio: {sharpeRatio:F4}"
                      logInfo $"Total Return: {totalReturn:P2}"
                      logInfo $"Annualized Return: {annualizedReturn:P2}"
                      logInfo $"Max Drawdown: {maxDrawdown:P2}"

                      // Charts are already generated above
                      logInfo "Analysis complete"

                      return Ok ()
                else
                  logInfo $"Training {config.MLAlgorithm} model..."

                  let model =
                    MachineLearning.trainModel
                      mlContext
                      trainData
                      config.MLAlgorithm
                      config

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

                    let healthReport =
                      MachineLearning.assessModelHealth evaluation

                    logInfo (
                      sprintf "Model Health Status: %A" healthReport.Status
                    )

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
                    logInfo "Evaluating simple trading strategy..."

                    let (signals,
                         strategyReturns,
                         cumulativeReturns,
                         sharpeRatio) =
                      TradingStrategy.evaluateStrategy
                        predictions
                        actualPrices
                        config

                    logInfo (sprintf "Strategy Sharpe Ratio: %.4f" sharpeRatio)

                    // Perform walk-forward backtesting
                    logInfo "Performing walk-forward backtesting..."

                    let backtestTrainModel =
                      fun (data : GoldDataRecord[]) ->
                        MachineLearning.trainModel
                          mlContext
                          data
                          config.MLAlgorithm
                          config

                    let backtestResult =
                      TradingStrategy.performWalkForwardBacktest
                        processedRecords
                        100 // Initial training size
                        20 // Test window size
                        backtestTrainModel
                        config

                    logInfo (
                      sprintf
                        "Backtest Total Return: %.2f%%"
                        (backtestResult.TotalReturn * 100.0)
                    )

                    logInfo (
                      sprintf
                        "Backtest Annualized Return: %.2f%%"
                        (backtestResult.AnnualizedReturn * 100.0)
                    )

                    logInfo (
                      sprintf
                        "Backtest Sharpe Ratio: %.4f"
                        backtestResult.SharpeRatio
                    )

                    logInfo (
                      sprintf
                        "Backtest Max Drawdown: %.2f%%"
                        (backtestResult.MaxDrawdown * 100.0)
                    )

                    logInfo (
                      sprintf
                        "Backtest Win Rate: %.2f%%"
                        (backtestResult.WinRate * 100.0)
                    )

                    logInfo (
                      sprintf
                        "Backtest Profit Factor: %.2f"
                        backtestResult.ProfitFactor
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

                      logInfo (
                        sprintf "Predicted next price: %.2f" predictedPrice
                      )

                      logInfo (
                        sprintf "Trading recommendation: %s" recommendation
                      )

                      logInfo
                        "Gold Price Prediction System completed successfully"

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
    let mutable useEnsemble = false

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
        | "--ensemble" ->
          useEnsemble <- true
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

    // Create base configuration
    let baseConfig =
      if useSGE then
        logInfo "Using Shanghai Gold Exchange data provider"
        Configuration.getSGEConfig ()
      else
        if etfSymbol <> "518880" then
          logInfo $"Using custom ETF symbol: {etfSymbol}"
        else
          logInfo "Using default ETF data provider (GLD ETF)"

        Configuration.getETFConfig etfSymbol

    // Override ensemble setting
    if useEnsemble then
      logInfo "Using ensemble model (combining all algorithms)"
    else
      let algorithmName =
        match baseConfig.MLAlgorithm with
        | LinearRegression -> "LinearRegression"
        | FastTreeRegression _ -> "FastTreeRegression"
        | FastForestRegression _ -> "FastForestRegression"
        | OnlineGradientDescentRegression -> "OnlineGradientDescentRegression"

      logInfo $"Using single {algorithmName} algorithm"

    { baseConfig with UseEnsemble = useEnsemble }

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
    printfn "Configuration can be set via environment variables:"

    printfn
      "  GOLD_MACHINE_API_URL         API base URL (default: http://127.0.0.1:8080/api/public)"

    printfn "  GOLD_MACHINE_SYMBOL           Symbol to use (default: 518880)"

    printfn
      "  GOLD_MACHINE_START_DATE       Start date YYYYMMDD (default: 20000101)"

    printfn "  GOLD_MACHINE_TRAIN_RATIO      Training ratio 0-1 (default: 0.8)"
    printfn "  GOLD_MACHINE_RISK_FREE_RATE   Risk-free rate 0-1 (default: 0.02)"
    printfn "  GOLD_MACHINE_DATA_PROVIDER    ETF or SGE (default: ETF)"

    printfn
      "  GOLD_MACHINE_ALGORITHM        ML algorithm: LinearRegression, FastTree,"

    printfn
      "                                FastForest, OnlineGradientDescent (default: LinearRegression)"

    printfn
      "  GOLD_MACHINE_USE_ENSEMBLE     Use ensemble model combining all algorithms (default: false)"

    printfn
      "  GOLD_MACHINE_FASTTREE_TREES   FastTree number of trees (default: 100)"

    printfn
      "  GOLD_MACHINE_FASTTREE_LEAVES  FastTree number of leaves per tree (default: 20)"

    printfn
      "  GOLD_MACHINE_FASTTREE_MIN_EXAMPLES FastTree minimum examples per leaf (default: 10)"

    printfn
      "  GOLD_MACHINE_FASTTREE_LEARNING_RATE FastTree learning rate (default: 0.2)"

    printfn
      "  GOLD_MACHINE_FASTTREE_SHRINKAGE FastTree shrinkage (default: 0.1)"

    printfn
      "  GOLD_MACHINE_FASTFOREST_TREES FastForest number of trees (default: 100)"

    printfn
      "  GOLD_MACHINE_FASTFOREST_LEAVES FastForest number of leaves per tree (default: 20)"

    printfn
      "  GOLD_MACHINE_FASTFOREST_MIN_EXAMPLES FastForest minimum examples per leaf (default: 10)"

    printfn
      "  GOLD_MACHINE_FASTFOREST_SHRINKAGE FastForest shrinkage (default: 0.1)"

    printfn ""
    printfn "Command line options (override environment variables):"

    printfn
      "  --etf <symbol>    Use ETF data with custom symbol (default: 518880)"

    printfn "  --ensemble        Use ensemble model (combines all algorithms)"
    printfn "  sge               Use Shanghai Gold Exchange data"
    printfn "  (no args)         Use default configuration"
    printfn ""
    printfn "Examples:"

    printfn
      "  dotnet run                                # Use default configuration"

    printfn
      "  dotnet run --etf 159941                   # Use custom ETF symbol"

    printfn
      "  dotnet run sge                            # Use Shanghai Gold Exchange"

    printfn
      "  GOLD_MACHINE_SYMBOL=123456 dotnet run     # Set via environment variable"

    printfn "  dotnet run --ensemble                     # Use ensemble model"

    printfn
      "  GOLD_MACHINE_USE_ENSEMBLE=true dotnet run # Set ensemble via environment variable"

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
