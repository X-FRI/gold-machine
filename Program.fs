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

          // Acquire data from API
          logInfo "Acquiring gold price data from API..."

          match! DataAcquisition.acquireGoldData validatedConfig with
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

                // Generate predictions for test data
                logInfo "Generating price predictions..."

                let predictions =
                  MachineLearning.predictBatch validatedModel testInputs

                // Evaluate trading strategy
                logInfo "Evaluating trading strategy..."

                let (signals, strategyReturns, cumulativeReturns, sharpeRatio) =
                  TradingStrategy.evaluateStrategy
                    predictions
                    actualPrices
                    config

                logInfo (sprintf "Strategy Sharpe Ratio: %.4f" sharpeRatio)

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
  /// Main entry point for the Gold Price Prediction System.
  /// </summary>
  /// <param name="argv">Command line arguments (currently unused).</param>
  /// <returns>Exit code (0 for success, 1 for failure).</returns>
  [<EntryPoint>]
  let main argv =
    printfn "==================================="

    let config = Configuration.getDefaultConfig ()

    match Async.RunSynchronously (runPredictionWorkflow config) with
    | Ok _ ->
      printfn "Program completed successfully."
      0
    | Error err ->
      handleError err
      printfn "Program failed. Check logs above for details."
      1
