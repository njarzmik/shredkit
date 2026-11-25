"""Command line: `shredkit demo`, `shredkit synth-run DIR`, `shredkit rawrec RUN_DIR`."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .codec import b58encode
from .demo import run_demo
from .rawrec import BatchResult, RunStats, SlotResult, deshred_run, write_run
from .synth import make_key, synthetic_slot


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.func(args)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shredkit", description="Solana Merkle shreds: verify, recover, deshred.")
    sub = p.add_subparsers(required=True)

    d = sub.add_parser("demo", help="run the whole pipeline on a synthetic, signed slot")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--drop", type=int, default=32, help="shreds dropped from each 64-shred set (0..32)")
    d.set_defaults(func=_cmd_demo)

    s = sub.add_parser("synth-run", help="write a synthetic rawrec run directory (rec_ and pay_ files)")
    s.add_argument("dir", type=Path)
    s.add_argument("--slots", type=int, default=3)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--lose", type=float, default=0.0, help="fraction of shreds lost in transit (default 0)")
    s.set_defaults(func=_cmd_synth_run)

    r = sub.add_parser("rawrec", help="deshred a rawrec recording into entries and transactions")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--json", action="store_true", help="one JSON object per batch instead of the text summary")
    r.add_argument("--txs", action="store_true", help="also list every transaction's first signature")
    r.add_argument("--window", type=int, default=256, help="slots kept open for late payloads (default 256)")
    r.set_defaults(func=_cmd_rawrec)
    return p


def _cmd_demo(args: argparse.Namespace) -> int:
    ok = run_demo(seed=args.seed, drop=args.drop)
    print("all checks passed" if ok else "CHECK FAILED")
    return 0 if ok else 1


def _cmd_synth_run(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    leader = make_key(rng)
    payloads, n_tx = [], 0
    for k in range(args.slots):
        syn = synthetic_slot(rng, leader, 1_000 + k, n_batches=3, zero_padding={1: 500})
        payloads += [p for p in syn.payloads if rng.random() >= args.lose]
        n_tx += len(syn.transactions)
    write_run(args.dir, payloads)
    print(f"wrote {len(payloads)} shreds of {args.slots} slots ({n_tx} transactions) to {args.dir}")
    return 0


def _cmd_rawrec(args: argparse.Namespace) -> int:
    stats = RunStats()
    totals: Counter = Counter()
    try:
        for result in deshred_run(args.run_dir, window=args.window, stats=stats):
            _tally(result, totals)
            if args.json:
                for b in result.batches:
                    print(json.dumps(_batch_json(b)))
            else:
                print(_slot_line(result))
            if args.txs:
                _print_txs(result)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(_summary(totals, stats), file=sys.stderr if args.json else sys.stdout)
    return 0 if totals["errors"] == 0 and totals["trailing_bytes"] == 0 else 1


def _tally(result: SlotResult, totals: Counter) -> None:
    totals["slots"] += 1
    totals["incomplete"] += result.incomplete_batches
    for b in result.batches:
        totals["batches"] += 1
        if b.batch is None:
            totals["errors"] += 1
            continue
        totals[b.batch.status] += 1
        for t in b.batch.transactions:
            totals["tx_" + t.version] += 1


def _slot_line(result: SlotResult) -> str:
    statuses = Counter(b.batch.status if b.batch else "error" for b in result.batches)
    versions = Counter(t.version for b in result.batches if b.batch for t in b.batch.transactions)
    return (f"slot {result.slot}: {result.data_shreds} data shreds, {len(result.batches)} complete batches "
            f"{dict(sorted(statuses.items()))}, {result.incomplete_batches} incomplete, "
            f"txs {dict(sorted(versions.items()))}")


def _batch_json(b: BatchResult) -> dict:
    row = {"slot": b.slot, "first_index": b.first_index, "last_index": b.last_index}
    if b.batch is None:
        return row | {"status": "error", "error": b.error}
    versions = Counter(t.version for t in b.batch.transactions)
    return row | {"status": b.batch.status, "entries": len(b.batch.entries), "bytes": b.batch.size,
                  "tail_bytes": b.batch.tail_length, "marker_version": b.batch.marker_version,
                  "txs": dict(sorted(versions.items()))}


def _print_txs(result: SlotResult) -> None:
    for b in result.batches:
        if b.batch is None:
            continue
        for n, entry in enumerate(b.batch.entries):
            for t in entry.transactions:
                print(f"  {result.slot} batch@{b.first_index} entry {n} {t.version:6} {b58encode(t.first_signature)}")


def _summary(totals: Counter, stats: RunStats) -> str:
    txs = {k[3:]: v for k, v in sorted(totals.items()) if k.startswith("tx_")}
    kinds = {k: totals[k] for k in ("exact", "zero_padded", "marker", "trailing_bytes", "errors") if totals[k]}
    return (f"total: {totals['slots']} slots, {totals['batches']} complete batches {kinds}, "
            f"{totals['incomplete']} incomplete, txs {txs}; payload records {stats.payload_records}, "
            f"duplicates {stats.duplicate_payloads} (conflicting {stats.conflicting_payloads}), "
            f"late {stats.late_payloads}, torn bytes {stats.torn_bytes}")


if __name__ == "__main__":
    sys.exit(main())
