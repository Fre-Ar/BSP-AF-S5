# python/geodata/utils.py
import numpy as np, pandas as pd, os, pyarrow as pa, pyarrow.parquet as pq

POINTS_DTYPE = {
    "lon": np.float32, "lat": np.float32,
    "x": np.float32, "y": np.float32, "z": np.float32,
    "is_border": np.uint8, "r_band": np.uint8,
    "dkm_hint": np.float32, "id_a": np.int32, "id_b": np.int32,
}

def _save_points_parquet(path: str,
                         lon: np.ndarray, lat: np.ndarray, xyz: np.ndarray,
                         is_border: np.ndarray, r_band: np.ndarray,
                         dkm_hint: np.ndarray, id_a: np.ndarray, id_b: np.ndarray) -> str:
    """Write the *starting points only* (no labels) so other labelers can reuse them."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame({
        "lon": lon.astype(np.float32),
        "lat": lat.astype(np.float32),
        "x": xyz[:, 0].astype(np.float32),
        "y": xyz[:, 1].astype(np.float32),
        "z": xyz[:, 2].astype(np.float32),
        "is_border": is_border.astype(np.uint8),
        "r_band": r_band.astype(np.uint8),
        "dkm_hint": dkm_hint.astype(np.float32),
        "id_a": id_a.astype(np.int32),
        "id_b": id_b.astype(np.int32),
    })
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    return os.path.abspath(path)

def _load_points_parquet(path: str):
    """Read starting points and return the same arrays we generate in the sampler step."""
    table = pq.read_table(path)
    # Let Arrow choose sensible NumPy dtypes; then enforce exactly what we expect.
    df = table.to_pandas()  # <-- no types_mapper

    # enforce expected dtypes just in case
    for k, dt in POINTS_DTYPE.items():
        if k in df:
            df[k] = df[k].astype(dt, copy=False)

    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    xyz = np.stack([df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()], axis=1)
    is_border = df["is_border"].to_numpy()
    r_band = df["r_band"].to_numpy()
    dkm_hint = df["dkm_hint"].to_numpy()
    id_a = df["id_a"].to_numpy()
    id_b = df["id_b"].to_numpy()
    return lon, lat, xyz, is_border, r_band, dkm_hint, id_a, id_b

