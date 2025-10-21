namespace GoldMachine

open System
open System.Net.Http
open System.Threading.Tasks
open System.Text.Json
open Newtonsoft.Json

/// <summary>
/// Data provider implementations for different gold price data sources.
/// Provides concrete implementations of the IDataProvider interface.
/// </summary>
module DataProviders =

  /// <summary>
  /// Data provider for gold ETF data from the original API.
  /// </summary>
  type ETFDataProvider () =
    interface IDataProvider with
      /// <summary>
      /// Gets the name of the ETF data provider.
      /// </summary>
      member this.Name = "Gold ETF Provider"

      /// <summary>
      /// Gets the type of the data provider.
      /// </summary>
      member this.ProviderType = ETFProvider

      /// <summary>
      /// Fetches raw ETF data from the API.
      /// </summary>
      /// <param name="config">Configuration containing API parameters.</param>
      /// <returns>Result containing ETF raw data or an error.</returns>
      member this.FetchRawData config =
        async {
          try
            use client = new HttpClient ()
            client.Timeout <- TimeSpan.FromSeconds (30.0)

            let url = Configuration.buildApiUrl config
            let! response = client.GetAsync (url) |> Async.AwaitTask
            response.EnsureSuccessStatusCode () |> ignore

            let! content =
              response.Content.ReadAsStringAsync () |> Async.AwaitTask

            let responseData =
              JsonConvert.DeserializeObject<ETFResponse> (content)

            return Ok (ETF responseData)
          with
          | :? HttpRequestException as ex ->
            return
              Error (DataAcquisitionFailed $"HTTP request failed: {ex.Message}")
          | :? TaskCanceledException ->
            return Error (DataAcquisitionFailed "Request timed out")
          | ex ->
            return
              Error (DataAcquisitionFailed $"Unexpected error: {ex.Message}")
        }

  /// <summary>
  /// Data provider for Shanghai Gold Exchange data.
  /// </summary>
  type SGEDataProvider () =
    interface IDataProvider with
      /// <summary>
      /// Gets the name of the SGE data provider.
      /// </summary>
      member this.Name = "Shanghai Gold Exchange Provider"

      /// <summary>
      /// Gets the type of the data provider.
      /// </summary>
      member this.ProviderType = SGEProvider

      /// <summary>
      /// Fetches raw SGE data from the API.
      /// </summary>
      /// <param name="config">Configuration containing API parameters.</param>
      /// <returns>Result containing SGE raw data or an error.</returns>
      member this.FetchRawData config =
        async {
          try
            use client = new HttpClient ()
            client.Timeout <- TimeSpan.FromSeconds (30.0)

            // Build SGE API URL - assuming the endpoint follows similar pattern
            let url =
              sprintf
                "%s/spot_hist_sge?symbol=%s"
                config.ApiBaseUrl
                config.Symbol

            let! response = client.GetAsync (url) |> Async.AwaitTask
            response.EnsureSuccessStatusCode () |> ignore

            let! content =
              response.Content.ReadAsStringAsync () |> Async.AwaitTask

            let responseData =
              JsonConvert.DeserializeObject<SGEResponse> (content)

            return Ok (SGE responseData)
          with
          | :? HttpRequestException as ex ->
            return
              Error (DataAcquisitionFailed $"HTTP request failed: {ex.Message}")
          | :? TaskCanceledException ->
            return Error (DataAcquisitionFailed "Request timed out")
          | ex ->
            return
              Error (DataAcquisitionFailed $"Unexpected error: {ex.Message}")
        }

  /// <summary>
  /// Factory for creating data providers based on configuration.
  /// </summary>
  module DataProviderFactory =

    /// <summary>
    /// Creates a data provider instance based on the specified provider type.
    /// </summary>
    /// <param name="providerType">The type of data provider to create.</param>
    /// <returns>The created data provider instance.</returns>
    let createProvider (providerType : DataProviderType) : IDataProvider =
      match providerType with
      | ETFProvider -> ETFDataProvider () :> IDataProvider
      | SGEProvider -> SGEDataProvider () :> IDataProvider

    /// <summary>
    /// Gets the data provider from configuration.
    /// </summary>
    /// <param name="config">Configuration containing provider type.</param>
    /// <returns>The configured data provider instance.</returns>
    let getProviderFromConfig (config : GoldMachineConfig) =
      createProvider config.DataProvider
