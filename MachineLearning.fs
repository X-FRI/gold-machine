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
  /// <returns>A trained GoldPredictionModel.</returns>
  let trainModel
    (mlContext : MLContext)
    (trainingRecords : GoldDataRecord[])
    (algorithm : MLAlgorithm)
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
      | FastTreeRegression ->
        (EstimatorChain()
          .Append(mlContext.Transforms.Concatenate ("Features", "MA3", "MA9"))
          .Append(mlContext.Regression.Trainers.FastTree ())
          .Fit (trainData))
        :> ITransformer
      | FastForestRegression ->
        (EstimatorChain()
          .Append(mlContext.Transforms.Concatenate ("Features", "MA3", "MA9"))
          .Append(mlContext.Regression.Trainers.FastForest ())
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
  /// <returns>A trained GoldPredictionModel.</returns>
  let trainLinearRegression
    (mlContext : MLContext)
    (trainingRecords : GoldDataRecord[])
    =
    trainModel mlContext trainingRecords LinearRegression

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
  /// <param name="k">Number of folds (default 5).</param>
  /// <returns>Cross-validation results containing average metrics.</returns>
  let crossValidateModel
    (mlContext : MLContext)
    (trainingRecords : GoldDataRecord[])
    (algorithm : MLAlgorithm)
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

      let model = trainModel mlContext trainData algorithm

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
  /// <returns>The best algorithm and its cross-validation results.</returns>
  let selectBestAlgorithm
    (mlContext : MLContext)
    (trainingRecords : GoldDataRecord[])
    (algorithms : MLAlgorithm list)
    =
    let results =
      algorithms
      |> List.map (fun alg ->
        let cvResult = crossValidateModel mlContext trainingRecords alg 5
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
