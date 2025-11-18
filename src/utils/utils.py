# src/utils/utils.py
import numpy as np
import pandas as pd
import os
import json
import pyarrow as pa, pyarrow.parquet as pq
import torch

# --------------------------- math -------------------------

def _safe_norm(v: np.ndarray, axis=1, keepdims=True, eps=1e-15):
    n = np.linalg.norm(v, axis=axis, keepdims=keepdims)
    # Clamp very small norms up to eps so downstream division never sees 0
    return np.where(n < eps, eps, n)

def _safe_div(v: np.ndarray, n: np.ndarray, eps=1e-15):
    # Replace tiny denominators by 1.0 to avoid huge values / NaNs.
    n = np.where(n < eps, 1.0, n)
    return v / n

def _slerp(a_u, b_u, t):
    """
    Spherical linear interpolation (SLERP) between two unit vectors a_u, b_u.

    Parameters
    ----------
    a_u : ndarray
        Start unit vector, shape (3,) or (..., 3).
    b_u : ndarray
        End unit vector, same shape as `a_u`.
    t : float
        Interpolation parameter in [0, 1]. t=0 -> a_u, t=1 -> b_u.

    Returns
    -------
    v : ndarray
        Interpolated unit vector on the great circle segment between a_u and b_u.
    """
    dot = np.clip(np.dot(a_u, b_u), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-15:
        return a_u.copy()
    s = np.sin(theta)
    
    v = (np.sin((1.0 - t) * theta) / s) * a_u + (np.sin(t * theta) / s) * b_u
    return v

# --------------------------- concat utility -------------------------

def _concat_parquet_shards(
        shard_paths: list[str],
        out_path: str,
        compression: str = "zstd",
        row_group_size: int | None = 512_000
    ):
    """
    Concatenates several Parquet shard files into a single Parquet file.

    This reads tables shard-by-shard and row-group-by-row-group, so memory
    usage stays bounded even for large datasets.

    Parameters
    ----------
    shard_paths : list[str]
        Paths to individual Parquet shards (same schema).
    out_path : str
        Output Parquet path.
    compression : str, optional
        Compression codec for the output file (default "zstd").
    row_group_size : int or None, optional
        Target row group size passed to `write_table`. If None, the default
        of pyarrow is used.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    first_schema = None
    writer = None
    try:
        for p in shard_paths:
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                if first_schema is None:
                    # Initialize writer lazily with the schema of the first row group
                    first_schema = table.schema
                    writer = pq.ParquetWriter(out_path, first_schema, compression=compression)
                writer.write_table(table, row_group_size=row_group_size)
            del pf
    finally:
        if writer is not None:
            writer.close()
            
# ---------------------------- I/O --------------------------------

def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"

def write_json(out: str, dictionary: dict, name = ""):
    """
    Writes a dictionary as pretty-printed JSON, or print it to stdout.

    Parameters
    ----------
    out : str
        Output file path. If falsy (e.g., ""), the JSON is printed instead.
    dictionary : dict
        Serializable mapping to store.
    name : str, optional
        Human-readable name used in the success message.
    """
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)
        print(f"Wrote {name} JSON to {out}")
    else:
        print(json.dumps(dictionary, ensure_ascii=False, indent=2))
   
# --------------------- Pretty Strings ----------------------------

def human_int(n: int) -> str:
    """
    Converts an integer into a compact human-readable string.

    Examples
    --------
    10_000_000 -> "10M"
    1_000_000  -> "1M"
    120_000    -> "120k"
    999        -> "999"

    Parameters
    ----------
    n : int
        Integer to format.

    Returns
    -------
    str
        Human-readable representation with suffix ('k', 'M', 'B') when applicable.
    """
    for suffix, factor in (("B", 10**9), ("M", 10**6), ("k", 10**3)):
        if abs(n) >= factor:
            return f"{int(n // factor)}{suffix}"
    return str(n)
