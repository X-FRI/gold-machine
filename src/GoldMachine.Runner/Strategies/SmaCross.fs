module GoldMachine.Strategies.SmaCross

open System
open GoldMachine.Data

type Signal =
    | Buy
    | Sell
    | Hold

type State = {
    FastSma: decimal list
    SlowSma: decimal list
    InPosition: bool
}

let private sma period (data: decimal list) =
    if data.Length < period then None
    else Some (data |> List.take period |> List.average)

let private crossAbove fast slow = fast > slow
let private crossBelow fast slow = fast < slow

let initialState = { FastSma = []; SlowSma = []; InPosition = false }

let update (state: State) (bar: TradeBarData) : State * Signal =
    let fastSma = bar.Close :: state.FastSma |> List.truncate 5
    let slowSma = bar.Close :: state.SlowSma |> List.truncate 20

    let signal =
        match sma 5 fastSma, sma 20 slowSma with
        | Some f, Some s when crossAbove f s && not state.InPosition -> Buy
        | Some f, Some s when crossBelow f s && state.InPosition -> Sell
        | _ -> Hold

    let inPosition =
        match signal with
        | Buy -> true
        | Sell -> false
        | Hold -> state.InPosition

    { FastSma = fastSma; SlowSma = slowSma; InPosition = inPosition }, signal
