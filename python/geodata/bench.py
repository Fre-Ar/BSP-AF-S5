import os, time, statistics, itertools, multiprocessing as mp
from datetime import datetime
from geodata import make_dataset_parallel

# --------- Config (edit here) ----------
N_TOTAL = 10_000
SHARDS_LIST = [32]
WORKERS_LIST = [None]
REPEATS = 1                        # how many times to run each combo (>=1)
BASE_OUT = "bench"                 # base folder
RUN_NAME_PREFIX = "run"            # used in output folder names
SEED = None                        # keep None to include randomness in sampling
# -------------------------------------


def bench_once(n, shards_list, workers_list, base_out, run_name_prefix, seed):
    """Run one pass over all (shards, workers) combos. Returns {(sh, wk): [times...]}."""
    results = {}
    combos = list(itertools.product(shards_list, workers_list))

    for (sh, wk) in combos:
        key = (sh, wk)
        results.setdefault(key, [])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = os.path.join(
            base_out,
            f"bench_{run_name_prefix}_{sh}_{'auto' if wk is None else wk}_{ts}"
        )
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.perf_counter()
        make_dataset_parallel(
            n_total=n,
            out_dir=out_dir,
            shards_per_total=sh,
            max_workers=wk,   # None -> auto (CPU-1)
            seed=seed,
        )
        dt = time.perf_counter() - t0
        results[key].append(dt)

    return results


def summarize(all_runs):
    """Merge run dicts and compute medians."""
    merged = {}
    for run in all_runs:
        for key, times in run.items():
            merged.setdefault(key, [])
            merged[key].extend(times)

    summary = {}
    for key, times in merged.items():
        summary[key] = {
            "times": times,
            "median": statistics.median(times) if times else float("nan"),
        }
    return summary


def print_summary(summary):
    rows = []
    for (sh, wk), info in summary.items():
        rows.append((info["median"], sh, wk, info["times"]))
    rows.sort(key=lambda r: r[0])

    print("\n=== Benchmark Summary (sorted by median seconds) ===")
    for med, sh, wk, times in rows:
        wk_str = "auto" if wk is None else str(wk)
        times_str = ", ".join(f"{t:.2f}" for t in times)
        print(f"shards={sh:>3}  workers={wk_str:>4}  median={med:8.2f}s   runs=[{times_str}]")
    print("====================================================\n")


def main():
    os.makedirs(BASE_OUT, exist_ok=True)

    all_runs = []
    for rep in range(REPEATS):
        print(f"\n=== Repetition {rep+1}/{REPEATS} ===")
        run_name = f"{RUN_NAME_PREFIX}_r{rep+1}"
        res = bench_once(
            n=N_TOTAL,
            shards_list=SHARDS_LIST,
            workers_list=WORKERS_LIST,
            base_out=BASE_OUT,
            run_name_prefix=run_name,
            seed=SEED,
        )
        all_runs.append(res)

    summary = summarize(all_runs)
    print_summary(summary)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()