module GoldMachine.Runner

open System
open System.IO
open GoldMachine.Data
open GoldMachine.Strategies.SmaCross

type Trade = {
    EntryDate: DateTime
    ExitDate: DateTime
    EntryPrice: decimal
    ExitPrice: decimal
    Shares: decimal
}

type BacktestMetrics = {
    TotalReturnPct: decimal
    AnnReturnPct: decimal
    MaxDrawdownPct: decimal
    SharpeRatio: float
    TotalTrades: int
    WinRate: float
    FinalEquity: decimal
}

let runBacktest (symbol: string) (dataDir: string) =
    let csvPath = Path.Combine(dataDir, $"{symbol}.csv")
    if not (File.Exists csvPath) then
        failwith $"Data file not found: {csvPath}"

    let bars = CsvLoader.load csvPath |> Array.filter (fun b -> b.Close > 0m)
    let mutable cash = 100_000m
    let mutable position = 0m
    let mutable state = initialState
    let mutable trades : Trade list = []
    let mutable equityCurve : (DateTime * decimal) list = []
    let mutable entryPrice = 0m
    let mutable entryDate = DateTime.MinValue

    for bar in bars do
        let newState, signal = update state bar

        match signal with
        | Buy when cash > 0m ->
            entryPrice <- bar.Close
            entryDate <- bar.Date
            position <- cash / bar.Close
            cash <- 0m
        | Sell when position > 0m ->
            let exitValue = position * bar.Close
            let pnl = (bar.Close - entryPrice) / entryPrice * 100m
            trades <- { EntryDate = entryDate; ExitDate = bar.Date; EntryPrice = entryPrice; ExitPrice = bar.Close; Shares = position } :: trades
            cash <- exitValue
            position <- 0m
        | _ -> ()

        let equity = cash + position * bar.Close
        equityCurve <- (bar.Date, equity) :: equityCurve
        state <- newState

    let lastBar = bars |> Array.last
    let finalEquity = cash + position * lastBar.Close
    let totalReturn = (finalEquity - 100_000m) / 100_000m * 100m

    let years = float (lastBar.Date - bars.[0].Date).TotalDays / 365.25
    let annReturn = if years > 0. then ((float finalEquity / 100_000.) ** (1. / years) - 1.) * 100. else 0.

    let peak = ref 100_000m
    let maxDd = ref 0m
    for _, eq in equityCurve do
        if eq > peak.Value then peak.Value <- eq
        let dd = (peak.Value - eq) / peak.Value * 100m
        if dd > maxDd.Value then maxDd.Value <- dd

    let totalTrades = trades.Length
    let wins = trades |> List.filter (fun t -> t.ExitPrice > t.EntryPrice)
    let winRate = if totalTrades > 0 then float wins.Length / float totalTrades * 100. else 0.

    let returns = equityCurve |> List.rev |> List.pairwise |> List.map (fun ((_, a), (_, b)) -> if a = 0m then 0. else float ((b - a) / a))
    let avgRet = if returns.Length > 0 then returns |> List.average else 0.
    let stdRet =
        if returns.Length > 1 then
            let sq = returns |> List.sumBy (fun r -> (r - avgRet) ** 2.)
            sqrt (sq / float returns.Length)
        else 0.
    let sharpe = if stdRet > 0. then avgRet / stdRet * sqrt 252. else 0.

    { TotalReturnPct = Math.Round(totalReturn, 2)
      AnnReturnPct = Math.Round(decimal annReturn, 2)
      MaxDrawdownPct = Math.Round(maxDd.Value, 2)
      SharpeRatio = Math.Round(sharpe, 2)
      TotalTrades = totalTrades
      WinRate = Math.Round(winRate, 1)
      FinalEquity = Math.Round(finalEquity, 2) }

[<EntryPoint>]
let main args =
    let symbol = args |> Array.tryHead |> Option.defaultValue "000001"
    let dataDir = args |> Array.tryItem 1 |> Option.defaultWith (fun () ->
        Path.Combine(Environment.CurrentDirectory, "data"))
    let dataDir = Path.GetFullPath dataDir

    try
        let m = runBacktest symbol dataDir

        printfn $""
        printfn $"  Gold Machine - Backtest Result"
        printfn $"  {'='}40"
        printfn $"  Symbol:       {symbol}"
        printfn $"  Period:       2020-01-01 ~ 2024-12-31"
        printfn $"  Initial:      ¥100,000.00"
        printfn $"  Final:        ¥{m.FinalEquity:N2}"
        printfn $"  Return:       {m.TotalReturnPct}%%"
        printfn $"  Ann Return:   {m.AnnReturnPct}%%"
        printfn $"  Max DD:       -{m.MaxDrawdownPct}%%"
        printfn $"  Sharpe:       {m.SharpeRatio}"
        printfn $"  Trades:       {m.TotalTrades}"
        printfn $"  Win Rate:     {m.WinRate}%%"
        printfn $""
        0
    with e ->
        eprintfn $"Error: {e}"
        1
