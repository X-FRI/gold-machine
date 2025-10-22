namespace GoldMachine

open Microsoft.ML
open Microsoft.ML.Data
open System

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
  /// Trains a linear regression model using the Stochastic Dual Coordinate Ascent (SDCA) algorithm.
  /// </summary>
  /// <param name="mlContext">The ML context to use for training.</param>
  /// <param name="trainingRecords">Array of training data records.</param>
  /// <returns>A trained GoldPredictionModel.</returns>
  let trainLinearRegression
    (mlContext : MLContext)
    (trainingRecords : GoldDataRecord[])
    =
    let trainingData =
      trainingRecords
      |> Array.map (fun r ->
        { MA3 = r.MA3
          MA9 = r.MA9
          Label = float32 r.Close })
      |> Array.toSeq

    let trainData = mlContext.Data.LoadFromEnumerable (trainingData)

    let pipeline =
      EstimatorChain()
        .Append(mlContext.Transforms.Concatenate ("Features", "MA3", "MA9"))
        .Append (mlContext.Regression.Trainers.Sdca ())

    let model = pipeline.Fit (trainData)

    { MLContext = mlContext
      Model = model
      Schema = trainData.Schema }

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
  /// Monitors model performance trends over time.
  /// </summary>
  /// <param name="recentEvaluations">Array of recent model evaluations (most recent first).</param>
  /// <returns>Health report based on trend analysis.</returns>
  let monitorModelTrends (recentEvaluations : ModelEvaluation[]) =
    if recentEvaluations.Length < 2 then
      assessModelHealth recentEvaluations.[0]
    else
      let latest = recentEvaluations.[0]
      let previous = recentEvaluations.[1]

      // Check for degradation trends
      let mapeIncrease = latest.MAPE - previous.MAPE
      let rmseIncrease = latest.RMSE - previous.RMSE

      let mutable status = Normal
      let mutable messages = []
      let mutable recommendations = []
      let mutable riskLevel = 0.0

      if mapeIncrease > 0.5 then
        status <- Degrading
        messages <- "MAPE is increasing, model may be degrading" :: messages

        recommendations <-
          "Monitor model performance changes" :: recommendations

        riskLevel <- max riskLevel 0.4

      if rmseIncrease > latest.MAE * 0.5f then
        status <- OutlierDetected

        messages <-
          "RMSE is increasing rapidly, there may be new outliers" :: messages

        recommendations <- "Check recent data quality" :: recommendations
        riskLevel <- max riskLevel 0.5

      // Sharp deterioration
      if latest.MAPE > previous.MAPE * 1.5 then
        status <- Critical
        messages <- "Model performance is急剧下降" :: messages

        recommendations <-
          "Suggest immediately retraining the model" :: recommendations

        riskLevel <- 1.0

      let baseHealth = assessModelHealth latest

      { Status =
          if status > baseHealth.Status then status else baseHealth.Status
        Message =
          if messages.Length > 0 then
            String.Join ("; ", messages)
          else
            baseHealth.Message
        Recommendations = recommendations @ baseHealth.Recommendations
        RiskLevel = max riskLevel baseHealth.RiskLevel }

  /// <summary>
  /// Creates an ensemble of models with weights based on their performance.
  /// </summary>
  /// <param name="models">Array of trained models.</param>
  /// <param name="evaluations">Array of model evaluations corresponding to the models.</param>
  /// <returns>ModelEnsemble with weighted models.</returns>
  let createModelEnsemble
    (models : GoldPredictionModel[])
    (evaluations : ModelEvaluation[])
    =
    if models.Length <> evaluations.Length then
      failwith "Models and evaluations arrays must have the same length"

    // Assign weights to models based on MAE: smaller MAE means higher weight
    let weights =
      evaluations
      |> Array.map (fun eval -> 1.0 / (1.0 + float eval.MAE))
      |> Array.map (fun w ->
        let totalWeight =
          evaluations |> Array.sumBy (fun eval -> 1.0 / (1.0 + float eval.MAE))

        w / totalWeight)

    let ensembleEvaluation =
      { RSquared =
          weights
          |> Array.mapi (fun i w -> w * float evaluations.[i].RSquared)
          |> Array.sum
          |> float32
        SharpeRatio =
          weights
          |> Array.mapi (fun i w -> w * evaluations.[i].SharpeRatio)
          |> Array.sum
        MAE =
          weights
          |> Array.mapi (fun i w -> w * float evaluations.[i].MAE)
          |> Array.sum
          |> float32
        RMSE =
          weights
          |> Array.mapi (fun i w -> w * float evaluations.[i].RMSE)
          |> Array.sum
          |> float32
        MAPE =
          weights
          |> Array.mapi (fun i w -> w * evaluations.[i].MAPE)
          |> Array.sum }

    { Models = models
      Weights = weights
      Evaluation = ensembleEvaluation }

  /// <summary>
  /// Makes predictions using an ensemble of models.
  /// </summary>
  /// <param name="ensemble">The model ensemble.</param>
  /// <param name="input">Prediction input.</param>
  /// <returns>Weighted ensemble prediction.</returns>
  let predictEnsemble (ensemble : ModelEnsemble) (input : PredictionInput) =
    let predictions =
      ensemble.Models
      |> Array.mapi (fun i model ->
        let pred = predict model input
        float pred * ensemble.Weights.[i])

    predictions |> Array.sum |> float32

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
