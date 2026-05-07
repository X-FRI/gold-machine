#!/usr/bin/env python3
"""
A 股日线行情下载工具

基于 akshare 库，从东方财富接口获取 A 股日线数据。
支持前复权、后复权、不复权三种价格调整方式。
数据输出为 CSV 格式，列名保持中文。

用法：
    python scripts/fetch/akshare_daily.py 000001              # 单只股票
    python scripts/fetch/akshare_daily.py 000001 600519       # 多只股票
    python scripts/fetch/akshare_daily.py --pool              # 下载常用标的池
    python scripts/fetch/akshare_daily.py 000001 --adj hfq    # 后复权
"""

import argparse
import os
import sys

import akshare as ak


def fetch_stock(
    symbol: str,
    adjust: str = "qfq",
    output_dir: str = "data",
    start_date: str = "19900101",
    end_date: str = "20500101",
) -> str:
    """
    下载单只 A 股的历史日线行情。

    参数
    ----------
    symbol : str
        股票代码，如 "000001"（平安银行）。
    adjust : str
        复权方式。可选值：
          - "qfq"：前复权（默认，推荐用于回测）
          - "hfq"：后复权
          - ""   ：不复权
    output_dir : str
        输出目录，CSV 文件将保存在此目录下。
    start_date, end_date : str
        日期范围，格式 "YYYYMMDD"。

    返回
    -------
    str
        保存的文件路径。如果无数据返回空字符串。

    说明
    -------
    akshare 的 stock_zh_a_hist 接口对应东方财富日线行情，
    包含前复权和后复权数据。前复权调整了历史价格以反映分红送股，
    使回测中的收益率计算更为准确。
    """
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    # 数据可能为空（停牌或代码不存在）
    if df.empty:
        print(f"  [警告] {symbol} 无数据返回", file=sys.stderr)
        return ""

    # 按时序排列，确保回测方向正确
    df.sort_values("日期", inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{symbol}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  ✓ {len(df)} 条记录 → {path}")
    return path


def main():
    """CLI 入口：解析参数并执行下载。"""
    parser = argparse.ArgumentParser(
        description="下载 A 股日线行情（基于 akshare / 东方财富）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/fetch/akshare_daily.py 000001\n"
            "  python scripts/fetch/akshare_daily.py 000001 600519 --adj hfq\n"
            "  python scripts/fetch/akshare_daily.py --pool\n"
        ),
    )
    parser.add_argument("symbols", nargs="*", help="股票代码，如 000001 600519")
    parser.add_argument(
        "--pool",
        action="store_true",
        help="下载常用标的池（见 pool.py）",
    )
    parser.add_argument(
        "--adj",
        default="qfq",
        choices=["", "qfq", "hfq"],
        help="复权方式: qfq=前复权(默认), hfq=后复权, ''=不复权",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="CSV 输出目录（默认: data/）",
    )
    args = parser.parse_args()

    # 确定下载列表
    if args.pool:
        from pool import COMMON_POOL
        symbols = [code for code, _, _ in COMMON_POOL]
        print(f"下载常用标的池 ({len(symbols)} 只)...")
    elif args.symbols:
        symbols = args.symbols
    else:
        # 默认下载平安银行
        symbols = ["000001"]

    for sym in symbols:
        fetch_stock(sym, adjust=args.adj, output_dir=args.output)


if __name__ == "__main__":
    main()
