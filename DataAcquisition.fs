namespace GoldMachine

open System
open System.Net.Http
open System.Threading.Tasks
open System.Text.Json
open Newtonsoft.Json

/// <summary>
/// Data acquisition module for fetching gold ETF price data from external APIs.
/// Provides functionality to retrieve historical price data and convert it to structured records.
/// </summary>
module DataAcquisition =

  /// <summary>
  /// Fetches raw JSON data from the specified API endpoint.
  /// </summary>
  /// <param name="url">The API endpoint URL to fetch data from.</param>
  /// <returns>Result containing the JSON response string or an error.</returns>
  let fetchJsonData (url : string) =
    async {
      try
        use client = new HttpClient ()
        client.Timeout <- TimeSpan.FromSeconds (30.0)

        let! response = client.GetAsync (url) |> Async.AwaitTask
        response.EnsureSuccessStatusCode () |> ignore

        let! content = response.Content.ReadAsStringAsync () |> Async.AwaitTask
        return Ok content
      with
      | :? HttpRequestException as ex ->
        return
          Error (DataAcquisitionFailed $"HTTP request failed: {ex.Message}")
      | :? TaskCanceledException ->
        return Error (DataAcquisitionFailed "Request timed out")
      | ex ->
        return Error (DataAcquisitionFailed $"Unexpected error: {ex.Message}")
    }

  /// <summary>
  /// Parses JSON response string into an array of RawGoldData records.
  /// </summary>
  /// <param name="jsonString">The JSON string to parse.</param>
  /// <returns>Result containing array of RawGoldData or a parsing error.</returns>
  let parseGoldData (jsonString : string) =
    try
      let data = JsonConvert.DeserializeObject<RawGoldData[]> (jsonString)
      Ok data
    with ex ->
      Error (DataAcquisitionFailed $"JSON parsing failed: {ex.Message}")

  /// <summary>
  /// Converts raw API data to structured GoldDataRecord objects.
  /// Filters out invalid records and parses dates.
  /// </summary>
  /// <param name="rawData">Array of raw gold data from the API.</param>
  /// <returns>Result containing array of GoldDataRecord or conversion error.</returns>
  let convertToGoldDataRecords (rawData : RawGoldData[]) =
    try
      let records =
        rawData
        |> Array.filter (fun item ->
          not (String.IsNullOrWhiteSpace (item.Date)) && item.Close > 0.0)
        |> Array.map (fun item ->
          { Date = DateTime.Parse (item.Date)
            Close = item.Close
            MA3 = 0.0f // Will be calculated later
            MA9 = 0.0f }) // Will be calculated later
        |> Array.sortBy (fun r -> r.Date)

      if records.Length = 0 then
        Error (
          DataAcquisitionFailed "No valid data records found after filtering"
        )
      else
        Ok records
    with ex ->
      Error (DataAcquisitionFailed $"Data conversion failed: {ex.Message}")

  /// <summary>
  /// Fetches gold ETF data from the API and converts it to structured records.
  /// This is the main entry point for data acquisition.
  /// </summary>
  /// <param name="config">Configuration containing API parameters.</param>
  /// <returns>Result containing array of GoldDataRecord or an acquisition error.</returns>
  let acquireGoldData (config : GoldMachineConfig) =
    async {
      let url = Configuration.buildApiUrl config

      match! fetchJsonData url with
      | Error err -> return Error err
      | Ok jsonString ->
        match parseGoldData jsonString with
        | Error err -> return Error err
        | Ok rawData -> return convertToGoldDataRecords rawData
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
