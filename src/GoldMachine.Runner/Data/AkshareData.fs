namespace GoldMachine.Data

open System
open System.Globalization

type TradeBarData = {
    Date: DateTime
    Open: decimal
    High: decimal
    Low: decimal
    Close: decimal
    Volume: decimal
}

module CsvLoader =
    let load (path: string) : TradeBarData array =
        let lines = System.IO.File.ReadAllLines path
        lines.[1..]
        |> Array.map (fun line ->
            let p = line.Split(',')
            {
                Date = DateTime.ParseExact(p.[0], "yyyy-MM-dd", CultureInfo.InvariantCulture)
                Open = Decimal.Parse(p.[2])
                Close = Decimal.Parse(p.[3])
                High = Decimal.Parse(p.[4])
                Low = Decimal.Parse(p.[5])
                Volume = Decimal.Parse(p.[6])
            }
        )
