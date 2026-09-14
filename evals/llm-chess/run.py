"""LLM-chess CLI: Swiss-paired tournament, A/B-swap judging, Elo leaderboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from arena import load_prompts, run_swiss  # noqa: E402
from export import export_all  # noqa: E402
from judges import REGISTRY as JUDGES  # noqa: E402
from leaderboard import compute_ratings  # noqa: E402
from models import REGISTRY as MODELS  # noqa: E402

_console = Console(stderr=True)


def _parse_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--judge-versions expects judge=version, got: {item}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=HERE / "prompts.jsonl")
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=f"Model names. Available: {sorted(MODELS)}",
    )
    p.add_argument(
        "--judges",
        nargs="+",
        default=["random"],
        help=f"Judge names. Available: {sorted(JUDGES)}",
    )
    p.add_argument("--out", type=Path, default=HERE / "results" / "matches.jsonl")
    p.add_argument(
        "--cache",
        type=Path,
        default=HERE / "results" / "judge_cache.jsonl",
        help="Judge-call cache path. Use --no-cache to disable.",
    )
    p.add_argument("--no-cache", action="store_true", help="Disable the judge cache.")
    p.add_argument(
        "--response-cache",
        type=Path,
        default=HERE / "results" / "response_cache.jsonl",
        help="Per-(model, prompt) response cache. Use --no-response-cache to disable.",
    )
    p.add_argument(
        "--no-response-cache",
        action="store_true",
        help="Disable the response cache — every run regenerates from each model.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Swiss rounds per prompt. Capped at C(M,2) automatically.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for pairing tie-breaks.")
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread pool size for parallel model/judge calls.",
    )
    p.add_argument(
        "--judge-versions",
        nargs="+",
        default=None,
        metavar="JUDGE=VERSION",
        help="Filter leaderboard to specific judge versions.",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=200,
        help="Bootstrap resamples for 95% CIs. 0 disables.",
    )
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Where to write matches.csv, leaderboard.csv, head_to_head.csv. "
        "Default: same dir as --out.",
    )
    p.add_argument("--no-csv", action="store_true", help="Skip CSV export.")
    args = p.parse_args()

    models = [MODELS[n] for n in args.models]
    judges = [JUDGES[n] for n in args.judges]
    prompts = load_prompts(args.prompts)

    cache_path = None if args.no_cache else args.cache
    rcache_path = None if args.no_response_cache else args.response_cache

    _matches = run_swiss(
        models=models,
        judges=judges,
        prompts=prompts,
        out_path=args.out,
        rounds=args.rounds,
        cache_path=cache_path,
        response_cache_path=rcache_path,
        seed=args.seed,
        max_workers=args.workers,
    )

    jv_filter = _parse_kv(args.judge_versions) if args.judge_versions else None
    rows = compute_ratings(args.out, judge_versions=jv_filter, n_boot=args.n_boot)

    table = Table(title="Leaderboard (95% CI from bootstrap)", title_style="bold cyan")
    table.add_column("model", style="magenta")
    table.add_column("elo", justify="right", style="bold")
    table.add_column("ci95", justify="right", style="dim")
    table.add_column("W", justify="right", style="green")
    table.add_column("L", justify="right", style="red")
    table.add_column("T", justify="right", style="yellow")
    table.add_column("N", justify="right")
    for row in rows:
        table.add_row(
            row.model,
            f"{row.rating:.1f}",
            f"[{row.ci_low:.0f}, {row.ci_high:.0f}]",
            str(row.wins),
            str(row.losses),
            str(row.draws),
            str(row.games),
        )
    _console.print(table)
    _console.log(f"[dim]matches →[/] {args.out}")
    if cache_path is not None:
        _console.log(f"[dim]judge cache →[/] {cache_path}")

    if not args.no_csv:
        csv_dir = args.csv_dir or args.out.parent
        export_all(
            matches_path=args.out,
            out_dir=csv_dir,
            n_boot=args.n_boot,
            judge_versions=jv_filter,
        )


if __name__ == "__main__":
    main()
