from nuked_geodata import make_dataset_parallel as old_labeler
from sampler import make_dataset_parallel as new_labeler
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FOLDER_PATH = "python/geodata/parquet"

START_POINTS = os.path.join(FOLDER_PATH, "start_points_10k.parquet")
OLD_PATH     = os.path.join(FOLDER_PATH, "nuked.parquet")
NEW_PATH     = os.path.join(FOLDER_PATH, "sampled.parquet")
OUT_PREFIX   = os.path.join(FOLDER_PATH, "problem_points_10k_v2")

KEY_COLS = ["lon","lat","x","y","z","is_border","r_band"]

THRESHOLDS = [10.0, 25.0, 50.0, 100.0]  # km

def _read_parquet_df(path: str, cols=None) -> pd.DataFrame:
    table = pq.read_table(path, columns=cols)
    # IMPORTANT: don't use types_mapper here; let pandas choose defaults
    return table.to_pandas()

def _ensure_pt_id(df: pd.DataFrame) -> pd.DataFrame:
    if "pt_id" not in df.columns:
        df = df.copy()
        df["pt_id"] = np.arange(len(df), dtype=np.int64)
    return df

def _coerce_float32(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(np.float32, copy=False)
    return out

def _align_on_start_points(start_path: str, old_path: str, new_path: str) -> pd.DataFrame:
    # load start points and add a stable pt_id
    sp = _ensure_pt_id(_read_parquet_df(start_path))
    sp_keys = sp[["pt_id"] + KEY_COLS].copy()

    # load labeled datasets (only the needed cols)
    need = KEY_COLS + ["dist_km","c1_id","c2_id"]
    old_df = _read_parquet_df(old_path, cols=need).copy()
    new_df = _read_parquet_df(new_path, cols=need).copy()

    # normalize dtypes we’ll join on
    for df in (sp_keys, old_df, new_df):
        for k in ["lon","lat","x","y","z"]:
            df[k] = df[k].astype(np.float32, copy=False)
        df["is_border"] = df["is_border"].astype(np.uint8, copy=False)
        df["r_band"]    = df["r_band"].astype(np.uint8, copy=False)

    # try exact-merge first
    old_join = sp_keys.merge(old_df, on=KEY_COLS, how="left", suffixes=("", "_old"))
    new_join = sp_keys.merge(new_df, on=KEY_COLS, how="left", suffixes=("", "_new"))

    miss_old = int(old_join["dist_km"].isna().sum())
    miss_new = int(new_join["dist_km"].isna().sum())

    if miss_old or miss_new:
        # tolerant round-merge (6 d.p.) on coordinates only; keep is_border/r_band exact
        def add_rounds(df, cols, suffix):
            df = df.copy()
            for c in cols:
                df[f"{c}_{suffix}"] = df[c].round(6)
            return df

        sp_r   = add_rounds(sp_keys, ["lon","lat","x","y","z"], "r")
        old_r  = add_rounds(old_df,  ["lon","lat","x","y","z"], "r")
        new_r  = add_rounds(new_df,  ["lon","lat","x","y","z"], "r")
        rcols  = [f"{c}_r" for c in ["lon","lat","x","y","z"]]

        old_join = sp_r.merge(
            old_r, left_on=rcols+["is_border","r_band"], right_on=rcols+["is_border","r_band"], how="left"
        )
        new_join = sp_r.merge(
            new_r, left_on=rcols+["is_border","r_band"], right_on=rcols+["is_border","r_band"], how="left"
        )

        # restore original column names on outputs
        for df in (old_join, new_join):
            # drop rounding keys
            drop_cols = [c for c in df.columns if c.endswith("_r")]
            df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # rename so we can merge both sides by pt_id
    old_join = old_join.rename(columns={"dist_km":"dist_old","c1_id":"c1_old","c2_id":"c2_old"})
    new_join = new_join.rename(columns={"dist_km":"dist_new","c1_id":"c1_new","c2_id":"c2_new"})

    both = sp_keys[["pt_id"] + KEY_COLS] \
        .merge(old_join[["pt_id","dist_old","c1_old","c2_old"]], on="pt_id", how="left") \
        .merge(new_join[["pt_id","dist_new","c1_new","c2_new"]], on="pt_id", how="left")

    # coverage report
    n = len(sp_keys)
    mo = int(both["dist_old"].isna().sum())
    mn = int(both["dist_new"].isna().sum())
    if mo or mn:
        print(f"[WARN] unmatched rows — old:{mo} ({mo/n:.2%}), new:{mn} ({mn/n:.2%}). "
              "These are excluded from stats.")

    return both

def compare(old_path=OLD_PATH, new_path=NEW_PATH, start_path=START_POINTS, out_prefix=OUT_PREFIX):
    df = _align_on_start_points(start_path, old_path, new_path)
    valid = df.dropna(subset=["dist_old","dist_new"]).copy()

    valid["delta_km"] = (valid["dist_new"] - valid["dist_old"]).abs()
    valid["disagree_ids"] = (valid["c1_old"] != valid["c1_new"]) | (valid["c2_old"] != valid["c2_new"])

    # summary stats
    n = len(valid)
    n_dis = int(valid["disagree_ids"].sum())
    mean_d = float(valid["delta_km"].mean())
    med_d  = float(valid["delta_km"].median())
    max_d  = float(valid["delta_km"].max())

    print("\n=== Comparison Summary ===")
    print(f"Rows compared: {n:,}")
    print(f"ID disagreements: {n_dis:,} ({(n_dis/max(1,n))*100:.3f}%)")
    print(f"|Δdist_km| mean / median / max: {mean_d:.6f} / {med_d:.6f} / {max_d:.6f}")

    for th in THRESHOLDS:
        cnt = int((valid["delta_km"] >= th).sum())
        print(f"≥{int(th)} km: {cnt:,}")

    # save problem rows (≥ 10 km)
    problems = valid.loc[valid["delta_km"] >= 10.0, [
        "pt_id","lon","lat","x","y","z","is_border","r_band",
        "dist_old","dist_new","delta_km","c1_old","c2_old","c1_new","c2_new","disagree_ids"
    ]].sort_values("delta_km", ascending=False).reset_index(drop=True)

    if not problems.empty:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        pq_path  = out_prefix + ".parquet"
        csv_path = out_prefix + ".csv"
        problems.to_parquet(pq_path, index=False)
        problems.to_csv(csv_path, index=False)
        print(f"\nSaved {len(problems):,} problem points to:\n  {pq_path}\n  {csv_path}")
    else:
        print("\nNo points with |Δdist_km| ≥ 10 km. 🎉")

    return {
        "rows": n,
        "id_disagreements": n_dis,
        "mean_delta": mean_d,
        "median_delta": med_d,
        "max_delta": max_d,
        "threshold_counts": {int(th): int((valid['delta_km'] >= th).sum()) for th in THRESHOLDS},
        "problems_saved": int(len(problems)),
    }

def run():
    points = os.path.join(FOLDER_PATH, "start_points_10k.parquet")

    # 1) Generate points once (KD-tree or STRtree—either is fine)
    new_labeler(
        n_total=10_000,
        out_path=NEW_PATH,
        #export_points_path=points,   # write the shared starting points
        points_path=points,            # ensure we *generate* not load
        shards_per_total=32,
        #seed=37,
        shuffle_points=False,        # deterministic row order
    )

    # 2) Label the *exact same* points with the other implementation
    old_labeler(
        n_total=10_000,              # ignored when points_path is set; keep for logs
        out_path=OLD_PATH,
        points_path=points,          # reuse shared starting points
        shards_per_total=32,
        shuffle_points=False,        # keep the order identical
    )

if __name__ == "__main__":
    #import multiprocessing as mp
    #mp.set_start_method("spawn", force=True)
    #run()
    compare()