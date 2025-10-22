namespace GoldMachine

open Microsoft.ML
open Microsoft.ML.Data
open System
open Microsoft.ML.Trainers

/// <summary>
/// Machine learning module for training and evaluating gold price prediction models.
/// Provides functionality for model training, prediction, and evaluation.
/// </summary>
module MachineLearning =

  /// <summary>
  /// Creates a new ML context with a fixed seed for reproducible results.
  /// </summary>
  /// <returns>A configured MLContext instance.</returns>
  let createMLContext () = MLContext (seed = Nullable 0)

  /// <summary>
  /// Trains a machine learning model using the specified algorithm.
  /// </summary>
  /// <param name="mlContext">The ML context to use for training.</param>
  /// <param name="trainingRecords">Array of training data records.</param>
  /// <param name="algorithm">The algorithm to use for training.</param>
  /// <param name="config">Configuration containing algorithm parameters.</param>
  /// <returns>A trained GoldPredictionModel.</returns>
  let trainModel
    (mlContext : MLContext)
    (trainingRecords : GoldDataRecord[])
    (algorithm : MLAlgorithm)
    (config : GoldMachineConfig)
    =
    let trainingData =
      trainingRecords
      |> Array.map (fun r ->
        { MA3 = r.MA3
          MA9 = r.MA9
          Label = float32 r.Close })
      |> Array.toSeq

    let trainData = mlContext.Data.LoadFromEnumerable (trainingData)

    let model =
      match algorithm with
      | LinearRegression ->
        (EstimatorChain()
          .Append(mlContext.Transforms.Concatenate ("Features", "MA3", "MA9"))
          .Append(mlContext.Regression.Trainers.Sdca ())
          .Fit (trainData))
        :> ITransformer
      | FastTreeRegression fastTreeParams ->
        (EstimatorChain()
          .Append(mlContext.Transforms.Concatenate ("Features", "MA3", "MA9"))
          .Append(
            mlContext.Regression.Trainers.FastTree (
              numberOfTrees = fastTreeParams.NumberOfTrees,
              numberOfLeaves = fastTreeParams.NumberOfLeaves,
              learningRate = float fastTreeParams.LearningRate
            )
          )
          .Fit (trainData))
        :> ITransformer
      | FastForestRegression fastForestParams ->
        (EstimatorChain()
          .Append(mlContext.Transforms.Concatenate ("Features", "MA3", "MA9"))
          .Append(
            mlContext.Regression.Trainers.FastForest (
              numberOfTrees = fastForestParams.NumberOfTrees,
              numberOfLeaves = fastForestParams.NumberOfLeaves
            )
          )
          .Fit (trainData))
        :> ITransformer
      | OnlineGradientDescentRegression ->
        (EstimatorChain()
          .Append(mlContext.Transforms.Concatenate ("Features", "MA3", "MA9"))
          .Append(mlContext.Regression.Trainers.OnlineGradientDescent ())
          .Fit (trainData))
        :> ITransformer

    { MLContext = mlContext
      Model = model
      Schema = trainData.Schema
      Algorithm = algorithm }

  /// <summary>
  /// Trains a linear regression model using the Stochastic Dual Coordinate Ascent (SDCA) algorithm.
  /// </summary>
  /// <param name="mlContext">The ML context to use for training.</param>
  /// <param name="trainingRecords">Array of training data records.</param>
  /// <param name="config">Configuration containing algorithm parameters.</param>
  /// <returns>A trained GoldPredictionModel.</returns>
  let trainLinearRegression
    (mlContext : MLContext)
    (trainingRecords : GoldDataRecord[])
    (config : GoldMachineConfig)
    =
    trainModel mlContext trainingRecords LinearRegression config

  /// <summary>
  /// Makes a price prediction using the trained model.
  /// </summary>
  /// <param name="model">The trained prediction model.</param>
  /// <param name="input">The prediction input containing features.</param>
  /// <returns>The predicted price as a float32.</returns>
  let predict (model : GoldPredictionModel) (input : PredictionInput) =
    let predictionEngine =
      model.MLContext.Model.CreatePredictionEngine<
        PredictionInput,
        PredictionOutput
       > (
        model.Model
      )

    predictionEngine.Predict(input).Score

  /// <summary>
  /// Makes predictions for a sequence of input data.
  /// </summary>
  /// <param name="model">The trained prediction model.</param>
  /// <param name="inputs">Sequence of prediction inputs.</param>
  /// <returns>Array of predicted prices.</returns>
  let predictBatch
    (model : GoldPredictionModel)
    (inputs : PredictionInput seq)
    =
    inputs |> Seq.map (predict model) |> Seq.toArray

  /// <summary>
  /// Calculates the R-squared coefficient of determination.
  /// </summary>
  /// <param name="actual">Array of actual values.</param>
  /// <param name="predicted">Array of predicted values.</param>
  /// <returns>The R-squared value between 0 and 1.</returns>
  let calculateRSquared (actual : float32[]) (predicted : float32[]) =
    match DataProcessing.validateArrayLengths [| actual ; predicted |] with
    | Error _ -> 0.0f
    | Ok _ ->
      let meanActual = Array.average actual
      let ssTot = actual |> Array.sumBy (fun x -> (x - meanActual) ** 2.0f)

      if ssTot = 0.0f then
        0.0f
      else
        let ssRes =
          Array.zip actual predicted
          |> Array.sumBy (fun (a, p) -> (a - p) ** 2.0f)

        1.0f - (ssRes / ssTot)

  /// <summary>
  /// Calculates the Mean Absolute Error (MAE).
  /// </summary>
  /// <param name="actual">Array of actual values.</param>
  /// <param name="predicted">Array of predicted values.</param>
  /// <returns>The mean absolute error.</returns>
  let calculateMAE (actual : float32[]) (predicted : float32[]) =
    match DataProcessing.validateArrayLengths [| actual ; predicted |] with
    | Error _ -> 0.0f
    | Ok _ ->
      Array.zip actual predicted |> Array.averageBy (fun (a, p) -> abs (a - p))

  /// <summary>
  /// Calculates the Root Mean Squared Error (RMSE).
  /// </summary>
  /// <param name="actual">Array of actual values.</param>
  /// <param name="predicted">Array of predicted values.</param>
  /// <returns>The root mean squared error.</returns>
  let calculateRMSE (actual : float32[]) (predicted : float32[]) =
    match DataProcessing.validateArrayLengths [| actual ; predicted |] with
    | Error _ -> 0.0f
    | Ok _ ->
      Array.zip actual predicted
      |> Array.averageBy (fun (a, p) -> (a - p) ** 2.0f)
      |> sqrt

  /// <summary>
  /// Calculates the Mean Absolute Percentage Error (MAPE).
  /// </summary>
  /// <param name="actual">Array of actual values.</param>
  /// <param name="predicted">Array of predicted values.</param>
  /// <returns>The mean absolute percentage error as a percentage.</returns>
  let calculateMAPE (actual : float32[]) (predicted : float32[]) =
    match DataProcessing.validateArrayLengths [| actual ; predicted |] with
    | Error _ -> 0.0
    | Ok _ ->
      Array.zip actual predicted
      |> Array.filter (fun (a, _) -> a <> 0.0f) // Avoid division by zero
      |> Array.averageBy (fun (a, p) -> float (abs ((a - p) / a)) * 100.0)

  /// <summary>
  /// Evaluates the performance of a trained model.
  /// </summary>
  /// <param name="model">The trained prediction model.</param>
  /// <param name="testInputs">Test input data for prediction.</param>
  /// <param name="actualPrices">Actual prices for comparison.</param>
  /// <returns>ModelEvaluation containing performance metrics.</returns>
  let evaluateModel
    (model : GoldPredictionModel)
    (testInputs : PredictionInput seq)
    (actualPrices : float32[])
    =
    let predictions = predictBatch model testInputs
    let rSquared = calculateRSquared actualPrices predictions
    let mae = calculateMAE actualPrices predictions
    let rmse = calculateRMSE actualPrices predictions
    let mape = calculateMAPE actualPrices predictions

    { RSquared = rSquared
      SharpeRatio = 0.0 // Sharpe ratio calculated separately in trading strategy
      MAE = mae
      RMSE = rmse
      MAPE = mape }

  /// <summary>
  /// Creates prediction input from a gold data record.
  /// </summary>
  /// <param name="record">The gold data record to convert.</param>
  /// <returns>Prediction input with moving averages as features.</returns>
  let createPredictionInput (record : GoldDataRecord) =
    { MA3 = record.MA3
      MA9 = record.MA9 }

  /// <summary>
  /// Assesses the health of a model based on its evaluation metrics.
  /// </summary>
  /// <param name="evaluation">The model evaluation metrics.</param>
  /// <returns>ModelHealthReport containing health assessment.</returns>
  let assessModelHealth (evaluation : ModelEvaluation) =
    let mutable status = Normal
    let mutable messages = []
    let mutable recommendations = []
    let mutable riskLevel = 0.0

    // MAPE-based degradation check
    if evaluation.MAPE > 2.0 then
      status <- Degrading
      messages <- "Model degradation, need to retrain" :: messages
      recommendations <- "Suggest retraining the model" :: recommendations

      recommendations <-
        "Check data quality and market conditions" :: recommendations

      riskLevel <- max riskLevel 0.7

    // RMSE vs MAE ratio check for outliers
    if evaluation.RMSE > evaluation.MAE * 2.0f then
      status <- OutlierDetected
      messages <- "There are outliers, need to check data quality" :: messages
      recommendations <- "Check for outliers in the data" :: recommendations

      recommendations <-
        "Consider using robust statistical methods" :: recommendations

      riskLevel <- max riskLevel 0.6

    // Additional checks
    if evaluation.RSquared < 0.1f then
      status <- Critical
      messages <- "Model explanation power is too low" :: messages

      recommendations <-
        "Consider replacing the model algorithm or feature engineering"
        :: recommendations

      riskLevel <- max riskLevel 0.9

    if evaluation.MAPE > 5.0 then
      status <- Critical
      messages <- "Prediction error is too high, need to stop using" :: messages

      recommendations <-
        "Stop using the model for trading decisions" :: recommendations

      riskLevel <- 1.0

    let finalStatus =
      if status = Normal && messages.Length > 0 then Degrading else status

    let message = String.Join ("; ", messages)

    let defaultMessage =
      if finalStatus = Normal then "Model performance is normal" else message

    { Status = finalStatus
      Message = defaultMessage
      Recommendations = recommendations
      RiskLevel = riskLevel }

  /// <summary>
  /// Estimates prediction intervals using RMSE and model uncertainty.
  /// </summary>
  /// <param name="prediction">Point prediction.</param>
  /// <param name="rmse">Root Mean Squared Error of the model.</param>
  /// <param name="confidenceLevel">Confidence level (e.g., 0.95 for 95%).</param>
  /// <returns>Tuple of (lowerBound, upperBound) for the prediction interval.</returns>
  let estimatePredictionInterval
    (prediction : float32)
    (rmse : float32)
    (confidenceLevel : float)
    =
    // Use a simplified method: estimate based on RMSE standard deviation
    // In practice, you can use t-distribution or empirical methods
    let zScore =
      match confidenceLevel with
      | 0.95 -> 1.96
      | 0.99 -> 2.576
      | 0.90 -> 1.645
      | _ -> 1.96 // Default 95% confidence interval

    let margin = float rmse * zScore
    let predFloat = float prediction

    (predFloat - margin, predFloat + margin)

  /// <summary>
  /// Adjusts prediction intervals based on MAPE for different price ranges.
  /// </summary>
  /// <param name="baseInterval">Base prediction interval.</param>
  /// <param name="mape">Mean Absolute Percentage Error.</param>
  /// <param name="currentPrice">Current price for percentage adjustment.</param>
  /// <returns>Adjusted prediction interval.</returns>
  let adjustIntervalForPriceRange
    (baseInterval : float * float)
    (mape : float)
    (currentPrice : float)
    =
    let percentageAdjustment = mape / 100.0
    let priceBasedWidth = currentPrice * percentageAdjustment

    let lower, upper = baseInterval
    let center = (lower + upper) / 2.0
    let baseWidth = upper - lower

    // Adjust interval width based on MAPE
    let adjustedWidth = max baseWidth priceBasedWidth
    let adjustedLower = center - adjustedWidth / 2.0
    let adjustedUpper = center + adjustedWidth / 2.0

    (adjustedLower, adjustedUpper)

  /// <summary>
  /// Performs k-fold cross-validation on the training data.
  /// </summary>
  /// <param name="mlContext">The ML context to use.</param>
  /// <param name="trainingRecords">Training data records.</param>
  /// <param name="algorithm">Algorithm to evaluate.</param>
  /// <param name="config">Configuration containing algorithm parameters.</param>
  /// <param name="k">Number of folds (default 5).</param>
  /// <returns>Cross-validation results containing average metrics.</returns>
  let crossValidateModel
    (mlContext : MLContext)
    (trainingRecords : GoldDataRecord[])
    (algorithm : MLAlgorithm)
    (config : GoldMachineConfig)
    (k : int)
    =
    if k < 2 then failwith "Number of folds must be at least 2"

    if trainingRecords.Length < k then
      failwith "Not enough data for cross-validation"

    let foldSize = trainingRecords.Length / k
    let mutable results = []

    for fold in 0 .. k - 1 do
      let testStart = fold * foldSize

      let testEnd =
        if fold = k - 1 then
          trainingRecords.Length - 1
        else
          (fold + 1) * foldSize - 1

      let testData = trainingRecords.[testStart..testEnd]

      let trainData =
        if testStart = 0 then
          trainingRecords.[testEnd + 1 ..]
        elif testEnd = trainingRecords.Length - 1 then
          trainingRecords.[.. testStart - 1]
        else
          Array.concat
            [| trainingRecords.[.. testStart - 1]
               trainingRecords.[testEnd + 1 ..] |]

      let model = trainModel mlContext trainData algorithm config

      let testInputs =
        testData |> Array.map createPredictionInput |> Array.toSeq

      let actualPrices = testData |> Array.map (fun r -> float32 r.Close)

      let evaluation = evaluateModel model testInputs actualPrices
      results <- evaluation :: results

    // Calculate average metrics across folds
    let avgRSquared = results |> List.averageBy (fun e -> e.RSquared)
    let avgMAE = results |> List.averageBy (fun e -> e.MAE)
    let avgRMSE = results |> List.averageBy (fun e -> e.RMSE)
    let avgMAPE = results |> List.averageBy (fun e -> e.MAPE)

    { RSquared = avgRSquared
      MAE = avgMAE
      RMSE = avgRMSE
      MAPE = avgMAPE
      SharpeRatio = 0.0 } // Sharpe ratio not calculated for CV

  /// <summary>
  /// Selects the best algorithm using cross-validation.
  /// </summary>
  /// <param name="mlContext">The ML context to use.</param>
  /// <param name="trainingRecords">Training data records.</param>
  /// <param name="algorithms">List of algorithms to evaluate.</param>
  /// <param name="config">Configuration containing algorithm parameters.</param>
  /// <returns>The best algorithm and its cross-validation results.</returns>
  let selectBestAlgorithm
    (mlContext : MLContext)
    (trainingRecords : GoldDataRecord[])
    (algorithms : MLAlgorithm list)
    (config : GoldMachineConfig)
    =
    let results =
      algorithms
      |> List.map (fun alg ->
        let cvResult =
          crossValidateModel mlContext trainingRecords alg config 5

        alg, cvResult)

    // Select algorithm with best R² score (could be changed to other metrics)
    results |> List.maxBy (fun (_, result) -> result.RSquared)

  /// <summary>
  /// Validates that a trained model is ready for prediction.
  /// </summary>
  /// <param name="model">The model to validate.</param>
  /// <returns>Result indicating validation success or error.</returns>
  let validateModel (model : GoldPredictionModel) =
    try
      let testInput =
        { MA3 = 1.0f
          MA9 = 1.0f }

      predict model testInput |> ignore
      Ok model
    with ex ->
      Error (ModelTrainingFailed $"Model validation failed: {ex.Message}")

  /// <summary>
  /// Gets all available ML algorithms for ensemble training with default parameters.
  /// </summary>
  /// <param name="config">Configuration containing algorithm parameters.</param>
  /// <returns>List of all supported ML algorithms.</returns>
  let getAllAlgorithms (config : GoldMachineConfig) =
    [ LinearRegression
      FastTreeRegression config.FastTreeParams
      FastForestRegression config.FastForestParams
      OnlineGradientDescentRegression ]

  /// <summary>
  /// Calculates weights for ensemble models based on their cross-validation performance.
  /// Uses R-squared scores to determine relative weights, with higher performing models getting higher weights.
  /// </summary>
  /// <param name="mlContext">The ML context to use.</param>
  /// <param name="trainingRecords">Training data for cross-validation.</param>
  /// <param name="models">List of trained models to weight.</param>
  /// <returns>List of weights corresponding to each model.</returns>
  let calculateEnsembleWeights
    (mlContext : MLContext)
    (trainingRecords : GoldDataRecord[])
    (models : GoldPredictionModel list)
    =
    if models.Length = 0 then
      []
    else
      // Perform quick evaluation using a holdout set
      let trainSize = int (float trainingRecords.Length * 0.8)
      let trainData = trainingRecords.[.. trainSize - 1]
      let validationData = trainingRecords.[trainSize..]

      let testInputs =
        validationData |> Array.map createPredictionInput |> Array.toSeq

      let actualPrices = validationData |> Array.map (fun r -> float32 r.Close)

      let performances =
        models
        |> List.map (fun model ->
          let predictions = predictBatch model testInputs
          let rSquared = calculateRSquared actualPrices predictions
          // Ensure R-squared is non-negative for weighting
          max 0.0f rSquared)

      // Normalize weights so they sum to 1
      let totalPerformance = performances |> List.sum |> float

      if totalPerformance = 0.0 then
        // If all models have 0 R-squared, use equal weights
        models |> List.map (fun _ -> 1.0 / float models.Length)
      else
        performances |> List.map (fun p -> float p / totalPerformance)

  /// <summary>
  /// Trains an ensemble model using all available algorithms.
  /// Attempts to train all algorithms but skips those that fail.
  /// </summary>
  /// <param name="trainingRecords">Array of training data records.</param>
  /// <param name="config">Configuration containing algorithm parameters.</param>
  /// <returns>Result containing trained ensemble model or error.</returns>
  let trainEnsembleModel
    (trainingRecords : GoldDataRecord[])
    (config : GoldMachineConfig)
    =
    try
      let mlContext = createMLContext ()
      let algorithms = getAllAlgorithms config

      let modelsAndAlgorithms =
        algorithms
        |> List.choose (fun alg ->
          try
            let model = trainModel mlContext trainingRecords alg config

            match validateModel model with
            | Ok validModel -> Some (alg, validModel)
            | Error err ->
              printfn $"Warning: Failed to train {alg}: {err}"
              None
          with ex ->
            printfn $"Warning: Exception training {alg}: {ex.Message}"
            None)

      if modelsAndAlgorithms.Length = 0 then
        Error (
          ModelTrainingFailed "No algorithms could be trained successfully"
        )
      else
        let models = modelsAndAlgorithms |> List.map snd

        // Calculate weights based on cross-validation performance
        let weights = calculateEnsembleWeights mlContext trainingRecords models

        Ok
          { Models = models
            Weights = weights
            MLContext = mlContext }
    with ex ->
      Error (ModelTrainingFailed $"Ensemble training failed: {ex.Message}")

  /// <summary>
  /// Makes a prediction using the ensemble model with weighted averaging.
  /// </summary>
  /// <param name="ensemble">The trained ensemble model.</param>
  /// <param name="input">The prediction input containing features.</param>
  /// <returns>The weighted average predicted price.</returns>
  let predictWithEnsemble (ensemble : EnsembleModel) (input : PredictionInput) =
    let modelsLen = List.length ensemble.Models
    let weightsLen = List.length ensemble.Weights

    if modelsLen <> weightsLen then
      failwith "Model and weight counts don't match"

    let weightedPredictions =
      List.zip ensemble.Models ensemble.Weights
      |> List.map (fun (model, weight) ->
        let prediction = predict model input |> float
        prediction * weight)

    let totalWeight = List.sum ensemble.Weights

    if totalWeight = 0.0 then
      0.0f
    else
      float32 (List.sum weightedPredictions / totalWeight)

  /// <summary>
  /// Makes batch predictions using the ensemble model.
  /// </summary>
  /// <param name="ensemble">The trained ensemble model.</param>
  /// <param name="inputs">Sequence of prediction inputs.</param>
  /// <returns>Array of weighted average predicted prices.</returns>
  let predictBatchWithEnsemble
    (ensemble : EnsembleModel)
    (inputs : PredictionInput seq)
    =
    inputs |> Seq.map (predictWithEnsemble ensemble) |> Seq.toArray

  /// <summary>
  /// Evaluates the ensemble model's performance.
  /// </summary>
  /// <param name="ensemble">The trained ensemble model.</param>
  /// <param name="testInputs">Test input data for prediction.</param>
  /// <param name="actualPrices">Actual prices for comparison.</param>
  /// <returns>EnsembleEvaluation containing performance metrics.</returns>
  let evaluateEnsembleModel
    (ensemble : EnsembleModel)
    (testInputs : PredictionInput seq)
    (actualPrices : float32[])
    =
    let predictions = predictBatchWithEnsemble ensemble testInputs
    let ensembleRSquared = calculateRSquared actualPrices predictions
    let ensembleMAE = calculateMAE actualPrices predictions
    let ensembleRMSE = calculateRMSE actualPrices predictions
    let ensembleMAPE = calculateMAPE actualPrices predictions

    // Evaluate individual models for comparison
    let individualEvaluations =
      ensemble.Models
      |> List.map (fun model ->
        let modelPredictions = predictBatch model testInputs
        let evaluation = evaluateModel model testInputs actualPrices
        model.Algorithm, evaluation)

    { IndividualEvaluations = individualEvaluations
      EnsembleRSquared = ensembleRSquared
      EnsembleMAE = ensembleMAE
      EnsembleRMSE = ensembleRMSE
      EnsembleMAPE = ensembleMAPE
      SharpeRatio = 0.0 } // Sharpe ratio calculated separately in trading strategy

  /// <summary>
  /// Validates that an ensemble model is ready for prediction.
  /// </summary>
  /// <param name="ensemble">The ensemble model to validate.</param>
  /// <returns>Result indicating validation success or error.</returns>
  let validateEnsembleModel (ensemble : EnsembleModel) =
    try
      let testInput =
        { MA3 = 1.0f
          MA9 = 1.0f }

      predictWithEnsemble ensemble testInput |> ignore

      let modelsLength = List.length ensemble.Models
      let weightsLength = List.length ensemble.Weights

      if modelsLength <> weightsLength then
        Error (
          ModelTrainingFailed
            "Ensemble model validation failed: mismatched model and weight counts"
        )
      elif modelsLength = 0 then
        Error (
          ModelTrainingFailed
            "Ensemble model validation failed: no models in ensemble"
        )
      elif ensemble.Weights |> List.exists (fun w -> w < 0.0) then
        Error (
          ModelTrainingFailed
            "Ensemble model validation failed: negative weights detected"
        )
      else
        Ok ensemble
    with ex ->
      Error (
        ModelTrainingFailed $"Ensemble model validation failed: {ex.Message}"
      )
