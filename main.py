"""Command-line entry point.

    python main.py analyze AAPL
    python main.py backtest --tickers AAPL MSFT NVDA --start 2025-01-01 --end 2025-12-31
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DISCLAIMER = "For research and education only. Not financial advice."


def _require_api_key() -> None:
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise SystemExit("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your Gemini key.")


def _explain_failure(error: Exception) -> str:
    message = str(error)
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return ("Gemini rate limit hit even after retrying. On the free tier this usually means the "
                "daily request quota is used up; check https://ai.dev/rate-limit, wait, or lower "
                "GEMINI_REQUESTS_PER_MINUTE in .env.")
    if "API_KEY_INVALID" in message or "API key not valid" in message:
        return "Gemini rejected the API key. Check GOOGLE_API_KEY in .env."
    return f"{type(error).__name__}: {message.splitlines()[0][:300]}"


def cmd_analyze(args: argparse.Namespace) -> None:
    _require_api_key()
    from trading_agent.config import REQUESTS_PER_MINUTE
    from trading_agent.runner import analyze

    if 0 < REQUESTS_PER_MINUTE < 10:
        print(f"Analyzing {args.ticker.upper()} (rate-limited to {REQUESTS_PER_MINUTE} requests/min, "
              "so this takes a few minutes)...")
    try:
        result = analyze(args.ticker, as_of=args.as_of, horizon_days=args.horizon)
    except Exception as e:
        if args.debug:
            raise
        raise SystemExit(f"Analysis failed. {_explain_failure(e)}")
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
    analyze.add_argument("--debug", action="store_true", help="Show full error tracebacks")
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
    if not getattr(args, "debug", False):
        # ADK logs full tracebacks for failed model calls; the CLI reports errors itself.
        logging.getLogger("google_adk").setLevel(logging.CRITICAL)
    args.func(args)


if __name__ == "__main__":
    main()
