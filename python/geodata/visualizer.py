import pandas as pd, geopandas as gpd, numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from matplotlib.colors import hsv_to_rgb


def plot_parquet_points(parquet_path, lon_col='lon', lat_col='lat',
                        sample=10_000, alpha=0.6, s=2, figsize=(10,5),
                        color_by=None, cmap='viridis'):
    df = pd.read_parquet(parquet_path, columns=[lon_col, lat_col] + ([color_by] if color_by else []))
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=0)

    plt.figure(figsize=figsize)
    if color_by:
        sc = plt.scatter(df[lon_col], df[lat_col], s=s, alpha=alpha, c=df[color_by], cmap=cmap)
        plt.colorbar(sc, label=color_by)
    else:
        plt.scatter(df[lon_col], df[lat_col], s=s, alpha=alpha)

    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.title(f'{parquet_path} — {len(df):,} points')
    plt.xlim(-180, 180); plt.ylim(-90, 90)
    plt.tight_layout(); plt.show()

def _resolve_world_layer():
    """
    Try to get a Natural Earth world layer via the 'geodatasets' package.
    Returns a path or None if unavailable.
    """
    try:
        from geodatasets import get_path
        # Try a few plausible keys; first one that works wins.
        candidates = [
            "naturalearth.cultural.v10.countries",
            "naturalearth.cultural.v10.admin_0_countries",
            "naturalearth.land",  # last resort: just land polygons
        ]
        for key in candidates:
            try:
                return get_path(key)
            except Exception:
                pass
    except Exception:
        pass
    return None

def _prep_dataframe(parquet_path, lon, lat, color_by, sample, extra_cols=()):
    cols = [lon, lat]
    if color_by and color_by not in cols:
        cols.append(color_by)
    for c in extra_cols:
        if c not in cols:
            cols.append(c)
    df = pd.read_parquet(parquet_path, columns=cols)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=0)
    return df

def _is_integer_series(s: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(s) or (
        pd.api.types.is_float_dtype(s) and np.all(np.isfinite(s)) and np.allclose(s, np.round(s))
    )

def _hash_colors(ids, sat=0.85, val=0.95):
    """
    Deterministic per-ID RGBA colors using a splitmix64-like bit mixer.
    Works with NumPy 2.0+ (no copy=False). Accepts any array-like of ints.
    """
    # normalize to a contiguous int64 array, then reinterpret bits as uint64
    i64 = np.asarray(ids, dtype=np.int64)
    u = i64.view(np.uint64)

    # splitmix64-ish mixer (uniform-ish hues)
    x = u + np.uint64(0x9E3779B97F4A7C15)
    x ^= (x >> np.uint64(30)); x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= (x >> np.uint64(27)); x *= np.uint64(0x94D049BB133111EB)
    x ^= (x >> np.uint64(31))

    # hue in [0,1)
    h = (x / np.float64(2**64)).astype(np.float64)

    hsv = np.stack(
        [h, np.full_like(h, sat, dtype=float), np.full_like(h, val, dtype=float)],
        axis=1
    )
    rgb = hsv_to_rgb(hsv)  # (N,3)
    rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=float)], axis=1)
    return rgba

def _categorical_palette(unique_vals, sat=0.80, val=0.95):
    """
    Distinct palette for small sets using equispaced hues on HSV wheel.
    Returns: dict {value -> RGBA}
    """
    n = max(1, len(unique_vals))
    hues = np.linspace(0, 1, n, endpoint=False)
    hsv = np.stack([hues, np.full(n, sat), np.full(n, val)], axis=1)
    rgb = hsv_to_rgb(hsv)
    rgba = np.concatenate([rgb, np.ones((n, 1), dtype=float)], axis=1)
    return {val: rgba[i] for i, val in enumerate(unique_vals)}

def plot_geopandas(
    parquet_path,
    lon='lon',
    lat='lat',
    color_by: str | None = None,   # e.g., 'c1_id', 'c2_id', 'dist_km'
    color_mode: str = "auto",      # 'auto' | 'categorical' | 'hashed' | 'continuous'
    log_scale: bool = False,       # for continuous numeric columns (e.g., dist_km)
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
    sample=200_000,
    world_path: str | None = None,
    figsize=(11, 5),
    alpha=0.8,
    markersize=2,                  # larger default so colors are visible
    basemap_facecolor='whitesmoke',
    basemap_edgecolor='gray',
    basemap_linewidth=0.3,
):
    df = _prep_dataframe(parquet_path, lon, lat, color_by, sample)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon], df[lat]),
        crs="EPSG:4326",
    )

    # Basemap
    if world_path is None:
        world_path = _resolve_world_layer()

    fig, ax = plt.subplots(figsize=figsize)
    if world_path is not None:
        try:
            world = gpd.read_file(world_path)
            if world.crs is not None and world.crs.to_epsg() != 4326:
                world = world.to_crs(4326)
            world.plot(ax=ax, color=basemap_facecolor,
                       edgecolor=basemap_edgecolor, linewidth=basemap_linewidth, zorder=1)
        except Exception as e:
            print(f"[WARN] Failed to load basemap ({world_path}): {e}")

    # Simple case
    if not color_by or color_by not in gdf.columns:
        ax.scatter(gdf[lon], gdf[lat], s=markersize, alpha=alpha, color='tab:blue',
                   linewidths=0, zorder=3)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(f"{parquet_path} — {len(gdf):,} points")
        plt.tight_layout(); plt.show()
        return

    series = gdf[color_by]
    title_suffix = f" • colored by {color_by}"

    # Decide mode
    mode = color_mode
    if mode == "auto":
        if _is_integer_series(series):
            nunique = series.nunique(dropna=True)
            mode = "categorical" if nunique <= 20 else "hashed"
        else:
            mode = "continuous"
    if mode in ("categorical", "hashed") and not _is_integer_series(series):
        mode = "continuous"

    # Draw
    # --- categorical ---
    if mode == "categorical":
        uniques = np.sort(series.dropna().unique())
        palette = _categorical_palette(uniques)
        colors = series.map(palette).fillna((0, 0, 0, 0)).to_numpy()
        print(f"[debug] categorical uniques={len(uniques)}  colors.shape={colors.shape}")
        # apply alpha onto color array (so alpha affects actual facecolors)
        colors = colors.copy(); colors[:, 3] = alpha
        ax.scatter(gdf[lon], gdf[lat], s=max(3, markersize), facecolors=colors,
                edgecolors='none', linewidths=0, zorder=3)

    # --- hashed ---
    elif mode == "hashed":
        ids = pd.to_numeric(series, errors="coerce").fillna(-1).astype("int64").to_numpy()
        colors = _hash_colors(ids)          # (N,4)
        colors = colors.copy(); colors[:, 3] = alpha  # set alpha into the color array

        # sanity debug
        uniq = np.unique(ids)
        print(f"[debug] hashed: unique IDs in sample = {len(uniq)}")
        print(f"[debug] colors sample: {colors[:3]}")

        ax.scatter(gdf[lon], gdf[lat],
                s=max(3, markersize),
                facecolors=colors,
                edgecolors='none',
                linewidths=0,
                zorder=3)
        title_suffix += " (hashed IDs — HSV wheel)"

    else:
        # continuous
        import matplotlib.colors as mcolors
        vals = pd.to_numeric(series, errors='coerce').to_numpy()
        v = vals.copy()
        lo_q, hi_q = clip_quantiles
        lo = np.nanquantile(v, lo_q) if np.isfinite(v).any() else 0.0
        hi = np.nanquantile(v, hi_q) if np.isfinite(v).any() else 1.0
        v = np.clip(v, lo, hi)
        if log_scale:
            eps = max(1e-6, 1e-9 * (hi - lo))
            v = np.log(v + eps)
            lo, hi = np.nanmin(v), np.nanmax(v)
            cb_label = f"log({color_by})"
        else:
            cb_label = color_by

        cmap = cmaps.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=lo, vmax=hi)
        sc = ax.scatter(gdf[lon], gdf[lat], s=markersize, c=v, cmap=cmap,
                        norm=norm, edgecolors='none', zorder=3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(cb_label)
        title_suffix += f" (continuous{' • log' if log_scale else ''} • clipped {int(lo_q*100)}–{int(hi_q*100)}%)"

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{parquet_path} — {len(gdf):,} points{title_suffix}")
    plt.tight_layout()
    plt.show()



# usage
plot_geopandas("python/geodata/parquet/dataset_all.parquet",
               sample=None,
               color_by="c1_id",
               color_mode="hashed")
               #log_scale=True) 

# usage
#plot_parquet_points("python/geodata/parquet/dataset_all.parquet", color_by=None)  # or color_by='dist_km' etc.


