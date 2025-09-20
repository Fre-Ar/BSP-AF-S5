from geodata import *
import time
from numpy.random import default_rng

def bench(run_name, ns, *, mode="scaling", repeats=3, seed=37):
    results = []
    # constant settings
    max_workers = 8
    mixture = (0.70, 0.30)

    # warmup (discard)
    _ = make_dataset_parallel(
        n_total=100_000,
        out_dir=f"bench/bench_warm_{run_name}",
        seed=seed,
        max_workers=max_workers,
        mixture=mixture,
        chunk_size=200_000 if mode=="scaling" else 10_000,
        shards_per_split=None if mode=="scaling" else 1,
    )

    for n in ns:
        times = []
        for r in range(repeats):
            t0 = time.perf_counter()
            make_dataset_parallel(
                n_total=n,
                out_dir=f"bench/bench_{run_name}_{n}_{r}",
                seed=seed,
                max_workers=max_workers,
                mixture=mixture,
                chunk_size=200_000 if mode=="scaling" else 10_000,
                shards_per_split=None if mode=="scaling" else 1,
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results.append((n, sorted(times)[len(times)//2]))  # median
        print(f"{run_name} n={n:,} -> {results[-1][1]:.2f}s")
    return results

if __name__ == "__main__":
    overhead_ns = [300, 1_000, 3_000, 10_000, 30_000]
    scaling_ns  = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]

    overhead = bench("overhead", overhead_ns, mode="overhead")
    scaling  = bench("scaling",  scaling_ns,  mode="scaling")

    # Optional: dump to CSV to plot later
    import pandas as pd
    pd.DataFrame(overhead, columns=["n_total","seconds"]).to_csv("overhead.csv", index=False)
    pd.DataFrame(scaling,  columns=["n_total","seconds"]).to_csv("scaling.csv", index=False)
