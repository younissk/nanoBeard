"""Pick the cheapest usable vast.ai offer across several GPU types.

    uv run python -m nanobeard.vast_offers --board
    uv run python -m nanobeard.vast_offers --pick        # one line: id min_bid bid dph gpu

Why this is not a shell one-liner:

  * Prices move within minutes. Measured 2026-09-10, ten minutes apart, the
    RTX_5090 bid floor went $0.202 -> $0.333 while the 4090 sat at $0.20 — the
    "obviously best card" swapped twice. Any hardcoded GPU default is a snapshot
    that is already stale.
  * vast bills your *bid*, not the floor. Bidding a $0.40 cap when the floor is
    $0.20 donates the difference for nothing, so the bid has to be derived from
    the chosen offer's own `min_bid`.
  * `vastai search offers` warns on an unrecognised query field and then
    silently returns on-demand offers anyway. That warning must not be
    swallowed — it is the difference between a $0.20/hr and a $0.54/hr instance.

Ranking is by derived bid price alone. It deliberately does not weigh in a
guessed tokens/sec per card: run `make autobatch` on the box for that, and pin
the winner with GPU=<name> if the cheapest card turns out to be slow money.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_GPUS = ("RTX_4090", "RTX_5090", "RTX_3090")
DEFAULT_MAX_DPH = 0.40
DEFAULT_BID_MULTIPLIER = 1.15
# Unrecognised query fields are a warning, not an error, and the results that
# come back are on-demand. Treat it as fatal.
BAD_FIELD_MARKER = "Unrecognized field"


@dataclass(frozen=True)
class Offer:
    gpu: str
    offer_id: int
    machine_id: int
    dph: float
    min_bid: float
    reliability: float
    inet_down: float
    cuda: float

    def bid(self, multiplier: float, cap: float) -> float:
        """What to actually bid: just above the floor, never above the cap."""
        return round(min(self.min_bid * multiplier, cap), 4)


def build_query(gpu: str, max_dph: float, inet_down: int, reliability: float,
                cuda_vers: str, datacenter: bool) -> str:
    q = (
        f"gpu_name={gpu} num_gpus=1 dph_total<={max_dph} "
        f"inet_down>={inet_down} reliability>={reliability} cuda_vers>={cuda_vers}"
    )
    if datacenter:
        q += " datacenter=true"
    return q


def search(query: str, interruptible: bool = True) -> list[dict]:
    """Raw offers for one query. Raises if vast reports a bad field."""
    if not shutil.which("vastai"):
        raise RuntimeError("vastai not on PATH — `uv tool install vastai`")
    cmd = ["vastai", "search", "offers", query, "-o", "dph+", "--raw"]
    if interruptible:
        cmd.insert(3, "-i")  # the flag; `type=bid` in the query is NOT a field
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if BAD_FIELD_MARKER in (proc.stdout + proc.stderr):
        raise RuntimeError(
            f"vast rejected a query field (results would silently be on-demand):\n"
            f"  query: {query}\n  {proc.stdout.strip()[:200]}"
        )
    if proc.returncode != 0:
        raise RuntimeError(f"vastai search failed: {proc.stderr.strip()[:200]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []


def to_offers(gpu: str, raw: list[dict]) -> list[Offer]:
    out = []
    for o in raw:
        dph = o.get("dph_total")
        if not dph:
            continue
        out.append(
            Offer(
                gpu=gpu,
                offer_id=o["id"],
                machine_id=int(o.get("machine_id") or 0),
                dph=float(dph),
                # On-demand rows carry no min_bid; fall back so the row is still
                # comparable rather than silently dropped.
                min_bid=float(o.get("min_bid") or dph),
                reliability=float(o.get("reliability2") or 0.0),
                inet_down=float(o.get("inet_down") or 0.0),
                cuda=float(o.get("cuda_max_good") or 0.0),
            )
        )
    return out


def cheapest_per_gpu(gpus, exclude_machines: set[int] | None = None, **kw) -> list[Offer]:
    """Best offer for each GPU type, cheapest first. Missing types are skipped.

    `exclude_machines` exists because a broken host advertises many offers — one
    per GPU slot — so retrying after a failure picks the same dead machine again.
    Measured: three consecutive launches landed on the same machine_id, all
    refusing to start with "GPU error".
    """
    interruptible = kw.pop("interruptible", True)
    skip = exclude_machines or set()
    best = []
    for gpu in gpus:
        offers = [
            o for o in to_offers(gpu, search(build_query(gpu, **kw), interruptible))
            if o.machine_id not in skip
        ]
        if offers:
            best.append(min(offers, key=lambda o: o.min_bid))
    return sorted(best, key=lambda o: o.min_bid)


def render_board(offers: list[Offer], multiplier: float, cap: float) -> str:
    head = (f"{'gpu':<10}{'offer':>10}{'machine':>9}{'dph':>8}{'min_bid':>9}"
            f"{'bid':>8}{'rel':>7}{'inet':>7}{'cuda':>6}")
    lines = [head, "-" * len(head)]
    for o in offers:
        lines.append(
            f"{o.gpu:<10}{o.offer_id:>10}{o.machine_id:>9}{o.dph:>8.3f}{o.min_bid:>9.3f}"
            f"{o.bid(multiplier, cap):>8.3f}{o.reliability:>7.3f}"
            f"{o.inet_down:>7.0f}{o.cuda:>6.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpus", default=",".join(DEFAULT_GPUS))
    ap.add_argument("--max-dph", type=float, default=DEFAULT_MAX_DPH)
    ap.add_argument("--bid-multiplier", type=float, default=DEFAULT_BID_MULTIPLIER)
    ap.add_argument("--inet-down", type=int, default=200)
    ap.add_argument("--reliability", type=float, default=0.95)
    ap.add_argument("--cuda-vers", default="12.9")
    ap.add_argument("--datacenter", action="store_true",
                    help="Datacenter hosts only. Measured 2.3x more expensive on vast.")
    ap.add_argument("--on-demand", action="store_true", help="Skip interruptible bidding")
    ap.add_argument("--exclude-machines", default="",
                    help="Comma-separated machine_ids to skip (hosts known to be broken)")
    ap.add_argument("--board", action="store_true", help="Print the ranked board")
    ap.add_argument("--pick", action="store_true",
                    help="Print `id min_bid bid dph gpu` for the winner (for scripts)")
    args = ap.parse_args()

    try:
        offers = cheapest_per_gpu(
            [g.strip() for g in args.gpus.split(",") if g.strip()],
            max_dph=args.max_dph,
            inet_down=args.inet_down,
            reliability=args.reliability,
            cuda_vers=args.cuda_vers,
            datacenter=args.datacenter,
            interruptible=not args.on_demand,
            exclude_machines={
                int(m) for m in args.exclude_machines.split(",") if m.strip()
            },
        )
    except RuntimeError as e:
        sys.exit(str(e))

    if not offers:
        sys.exit(
            f"No offers for {args.gpus} under ${args.max_dph}/hr.\n"
            f"Raise --max-dph, drop --datacenter, or widen --gpus."
        )

    if args.pick:
        w = offers[0]
        print(f"{w.offer_id} {w.min_bid} {w.bid(args.bid_multiplier, args.max_dph)} "
              f"{w.dph} {w.gpu} {w.machine_id}")
        return
    print(render_board(offers, args.bid_multiplier, args.max_dph))


if __name__ == "__main__":
    main()
