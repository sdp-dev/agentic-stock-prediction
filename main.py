"""Command-line entry point.

    python main.py analyze AAPL
    python main.py backtest --tickers AAPL MSFT NVDA --start 2025-01-01 --end 2025-12-31
"""

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DISCLAIMER = "For research and education only. Not financial advice."


def _require_api_key() -> None:
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_USE_VERTEXAI")):
        raise SystemExit("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your Gemini key.")


def cmd_analyze(args: argparse.Namespace) -> None:
    _require_api_key()
    from trading_agent.runner import analyze

    result = analyze(args.ticker, as_of=args.as_of, horizon_days=args.horizon)
    rec = result.recommendation

    if args.verbose:
        for name, report in result.reports.items():
            print(f"\n── {name.replace('_', ' ')} ──\n{report.strip()}")
        print()

    print(f"{rec.ticker}: {rec.action}  (confidence {rec.confidence:.0%}, {args.horizon}-day horizon)")
    print(f"\n{rec.rationale}\n")
    print("Key factors:")
    for factor in rec.key_factors:
        print(f"  + {factor}")
    print("Risks:")
    for risk in rec.risks:
        print(f"  - {risk}")
    print(f"\n{DISCLAIMER}")


def cmd_backtest(args: argparse.Namespace) -> None:
    if not args.baseline_only:
        _require_api_key()
    from backtest.harness import build_report, run_backtest

    out = Path(args.out)
    results = asyncio.run(run_backtest(
        tickers=[t.upper() for t in args.tickers],
        start=args.start,
        end=args.end,
        every=args.every,
        horizon=args.horizon,
        out=out,
        delay=args.delay,
        baseline_only=args.baseline_only,
    ))
    report = build_report(results, args.horizon)
    report_path = out.with_name("report.md")
    report_path.write_text(report)
    print(f"\n{report}\nSaved {out} and {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic stock prediction with Google ADK")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Get a recommendation for one stock")
    analyze.add_argument("ticker")
    analyze.add_argument("--as-of", help="Analyze as of a past date (YYYY-MM-DD)")
    analyze.add_argument("--horizon", type=int, default=20, help="Holding period in trading days")
    analyze.add_argument("-v", "--verbose", action="store_true", help="Print each analyst's report")
    analyze.set_defaults(func=cmd_analyze)

    backtest = sub.add_parser("backtest", help="Score the agent's calls on past dates")
    backtest.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "JPM", "XOM"])
    backtest.add_argument("--start", default="2025-01-01")
    backtest.add_argument("--end", default="2025-12-31")
    backtest.add_argument("--every", type=int, default=20, help="Trading days between decisions")
    backtest.add_argument("--horizon", type=int, default=20, help="Trading days to score each call over")
    backtest.add_argument("--out", default="results/predictions.csv")
    backtest.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between calls (rate limits)")
    backtest.add_argument("--baseline-only", action="store_true", help="Skip the LLM; score the baselines only")
    backtest.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
