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
