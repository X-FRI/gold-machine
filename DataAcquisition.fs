namespace GoldMachine

/// <summary>
/// Data acquisition module for fetching gold price data from multiple external APIs.
/// Provides unified interface for different data sources through the provider pattern.
/// </summary>
module DataAcquisition =

  /// <summary>
  /// Acquires data from the specified data provider.
  /// This is the main entry point for data acquisition using the provider pattern.
  /// </summary>
  /// <param name="provider">The data provider to use for fetching data.</param>
  /// <param name="config">Configuration containing data source parameters.</param>
  /// <returns>Result containing array of GoldDataRecord or an acquisition error.</returns>
  let acquireGoldData (provider : IDataProvider) (config : GoldMachineConfig) =
    async {
      match! provider.FetchRawData config with
      | Error err -> return Error err
      | Ok rawData ->
        let records = DataProcessing.convertRawDataToRecords rawData
        return Ok records
    }

  /// <summary>
  /// Validates that the acquired data meets minimum requirements.
  /// </summary>
  /// <param name="records">The gold data records to validate.</param>
  /// <param name="minRecords">Minimum number of records required.</param>
  /// <returns>Result indicating validation success or failure.</returns>
  let validateData (records : GoldDataRecord[]) (minRecords : int) =
    if records.Length < minRecords then
      Error (
        DataAcquisitionFailed
          $"Insufficient data: {records.Length} records, minimum {minRecords} required"
      )
    elif records |> Array.exists (fun r -> r.Close <= 0.0) then
      Error (DataAcquisitionFailed "Data contains invalid price values")
    else
      Ok records

  /// <summary>
  /// Legacy function for backward compatibility.
  /// Uses the provider configured in the configuration.
  /// </summary>
  /// <param name="config">Configuration containing API parameters.</param>
  /// <returns>Result containing array of GoldDataRecord or an acquisition error.</returns>
  let acquireGoldDataLegacy (config : GoldMachineConfig) =
    let provider =
      DataProviders.DataProviderFactory.getProviderFromConfig config

    acquireGoldData provider config
