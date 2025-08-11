#!/usr/bin/env python3
"""
Simple PMM benchmark script.
Measures:
- Token creation throughput
- Recall latency
- Chain verification latency

Usage:
  python scripts/benchmarks.py --tokens 1000 --recalls 5 --model-path enhanced_pmm_model.json

Notes:
- Results depend on hardware, configuration, and model/provider settings.
"""
import argparse
import time
from statistics import mean
from pathlib import Path

from pmm.enhanced_manager import EnhancedSelfModelManager
from pmm.enhanced_model import NextStageConfig


def fmt_ms(ms: float) -> str:
    """Format milliseconds with higher precision and µs for very small values."""
    if ms >= 1.0:
        return f"{ms:.3f} ms"
    return f"{ms * 1000.0:.1f} µs"


def timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = (time.perf_counter() - t0) * 1000.0
    return out, dt


essay = (
    "Today I reflected on improving recall integrity and managing archival. "
    "I noted patterns around stability and growth in behavior."
)


def bench_token_creation(mgr: EnhancedSelfModelManager, n: int) -> float:
    print(f"[tokenize] Adding {n} events...")
    start = time.perf_counter()
    durations = []
    for i in range(1, n + 1):
        _, dt = timeit(mgr.add_event, summary=f"Benchmark event {i}: {essay}")
        durations.append(dt)
        if i % max(1, n // 10) == 0 or i == n:
            avg = mean(durations)
            elapsed = time.perf_counter() - start
            remaining_iters = n - i
            eta_sec = (avg / 1000.0) * remaining_iters
            print(
                f"  · {i}/{n} events | avg={fmt_ms(avg)} | elapsed={elapsed:.2f}s | ETA={eta_sec:.2f}s",
                flush=True,
            )
    return mean(durations)


def bench_recall_latency(mgr: EnhancedSelfModelManager, recalls: int) -> float:
    print(f"[recall] Running {recalls} recall queries...")
    start = time.perf_counter()
    durations = []
    for i in range(1, recalls + 1):
        _, dt = timeit(mgr.recall_memories, cue="recall integrity", max_results=5)
        durations.append(dt)
        if i % max(1, recalls // 5) == 0 or i == recalls:
            avg = mean(durations)
            elapsed = time.perf_counter() - start
            remaining_iters = recalls - i
            eta_sec = (avg / 1000.0) * remaining_iters
            print(
                f"  · {i}/{recalls} recalls | avg={fmt_ms(avg)} | elapsed={elapsed:.2f}s | ETA={eta_sec:.2f}s",
                flush=True,
            )
    return mean(durations)


def bench_chain_verification(mgr: EnhancedSelfModelManager) -> float:
    # Uses public API verify_integrity(), returns summary. Measure end-to-end time.
    _, dt = timeit(mgr.verify_integrity)
    return dt


def main():
    ap = argparse.ArgumentParser(
        description="PMM simple benchmarks (no heavy backends by default)"
    )
    ap.add_argument("--tokens", type=int, default=50, help="Number of events to add")
    ap.add_argument(
        "--recalls",
        type=int,
        default=0,
        help="Number of recall queries (0 skips recall)",
    )
    ap.add_argument(
        "--model-path", default="enhanced_pmm_model.json", help="Model file path"
    )
    ap.add_argument(
        "--fresh", action="store_true", help="Start from a fresh model file"
    )
    ap.add_argument(
        "--enable-recall",
        action="store_true",
        help="Enable recall engine (may load embeddings)",
    )
    ap.add_argument(
        "--enable-inference",
        action="store_true",
        help="Enable local/API inference (may load models)",
    )
    args = ap.parse_args()

    if args.fresh:
        mp = Path(args.model_path)
        if mp.exists():
            mp.unlink()

    # Disable heavy components by default to avoid pulling large models
    cfg = NextStageConfig(
        enable_memory_tokens=True,
        enable_archival=True,
        enable_recall=bool(args.enable_recall),
        enable_local_inference=bool(args.enable_inference),
        enable_integrity_checks=True,
    )

    mgr = EnhancedSelfModelManager(
        model_path=args.model_path, enable_next_stage=True, config=cfg
    )

    try:
        print("[start] PMM benchmark")
        print("[mode] Batch-saving enabled during token creation to reduce disk I/O")
        mgr.begin_batch()
        try:
            avg_create_ms = bench_token_creation(mgr, args.tokens)
        finally:
            mgr.end_batch()
        avg_recall_ms = 0.0
        if args.enable_recall and args.recalls > 0:
            avg_recall_ms = bench_recall_latency(mgr, args.recalls)
        print("[integrity] Verifying chain...")
        verify_ms = bench_chain_verification(mgr)
    except KeyboardInterrupt:
        print("[abort] Benchmark interrupted by user.")
        return
    except Exception as e:
        print(f"[error] {e}")
        return

    print("[done] PMM Benchmarks (illustrative):")
    print(
        f"  Token creation avg: {fmt_ms(avg_create_ms)}/event over {args.tokens} events"
    )
    if args.enable_recall and args.recalls > 0:
        print(
            f"  Recall latency avg: {fmt_ms(avg_recall_ms)} over {args.recalls} queries"
        )
    else:
        print("  Recall latency avg: (skipped)")
    print(f"  Verify integrity:   {fmt_ms(verify_ms)}")


if __name__ == "__main__":
    main()
