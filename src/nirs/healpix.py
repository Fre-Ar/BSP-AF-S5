# healpix.py
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from viz.visualizer import _resolve_world_layer

# ---------------------------
# HEALPix NESTED tiling scheme
# ---------------------------

# ---------- Morton interleave helpers (for up to 16-bit ix/iy; i.e., nside ≤ 65536) ----------
def _part1by1_16(u: torch.Tensor) -> torch.Tensor:
    '''
    Spread the lower 16 bits of u so bits occupy even positions: b15..b0 -> b15 0 b14 0 ... b0 0.
    Works on integer tensors (int32/int64). Returns same dtype/device.
    '''
    u = u & 0x0000_FFFF
    u = (u | (u << 8))  & 0x00FF_00FF
    u = (u | (u << 4))  & 0x0F0F_0F0F
    u = (u | (u << 2))  & 0x3333_3333
    u = (u | (u << 1))  & 0x5555_5555
    return u

def _morton2D(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
    '''Interleave bits of ix (even positions) and iy (odd positions).'''
    return _part1by1_16(ix) | (_part1by1_16(iy) << 1)

# ---------- main: batch (B,3) unit vectors -> HEALPix NESTED pixel ids (B,) ----------
@torch.no_grad()
def healpix_vec2pix_nest_batch(x: torch.Tensor, nside: int) -> torch.Tensor:
    '''
    Vectorized HEALPix NESTED ang2pix for unit vectors.

    Args
    ----
    x      : Float tensor (B,3) or (...,3), assumed unit vectors (we clamp z for safety).
    nside  : int power of two (typical HEALPix constraint). Supports up to 65536.

    Returns
    -------
    pix_id : Long tensor (...,) with ids in [0, 12*nside*nside).
    '''
    assert nside >= 1 and (nside & (nside - 1)) == 0, "nside must be a power of two"
    orig_shape = x.shape[:-1]
    x = x.reshape(-1, 3)

    # Extract and clamp
    xv, yv, zv = x[:, 0], x[:, 1], x[:, 2]
    zv = torch.clamp(zv, -1.0, 1.0)
    za = torch.abs(zv)

    # phi in [0, 2π)
    phi = torch.atan2(yv, xv)
    two_pi = 2.0 * math.pi
    phi = torch.where(phi < 0, phi + two_pi, phi)

    # Common intermediates
    tt = phi / (0.5 * math.pi)                 # in [0,4)
    z0 = 2.0 / 3.0
    eq_mask = (za <= z0)

    # ----- Equatorial branch -----
    # jp/jm as in the reference formulas
    jp_eq = torch.floor(nside * (0.5 + tt - 0.75 * zv)).to(torch.int64)
    jm_eq = torch.floor(nside * (0.5 + tt + 0.75 * zv)).to(torch.int64)

    ifp_eq = jp_eq // nside
    ifm_eq = jm_eq // nside

    face_eq = torch.where(
        ifp_eq == ifm_eq, (ifp_eq & 3) + 4,
        torch.where(ifp_eq < ifm_eq, (ifp_eq & 3), (ifm_eq & 3) + 8)
    )

    ix_eq = (jm_eq % nside).to(torch.int64)
    iy_eq = (nside - (jp_eq % nside) - 1).to(torch.int64)

    # ----- Polar branch -----
    ntt = torch.floor(tt).to(torch.int64)
    ntt = torch.where(ntt >= 4, torch.full_like(ntt, 3), ntt)
    tp  = tt - ntt.to(tt.dtype)
    tmp = torch.sqrt(3.0 * (1.0 - za))         # (0,1]

    jp_po = torch.floor(nside * tp * tmp).to(torch.int64)
    jm_po = torch.floor(nside * (1.0 - tp) * tmp).to(torch.int64)
    # clamp to [0, nside-1]
    jp_po = torch.clamp(jp_po, 0, nside - 1)
    jm_po = torch.clamp(jm_po, 0, nside - 1)

    north_mask = (zv >= 0.0)
    face_po = torch.where(north_mask, ntt, ntt + 8)
    ix_po   = torch.where(north_mask, (nside - jm_po - 1), jp_po).to(torch.int64)
    iy_po   = torch.where(north_mask, (nside - jp_po - 1), jm_po).to(torch.int64)

    # ----- Select per region -----
    face = torch.where(eq_mask, face_eq, face_po).to(torch.int64)
    ix   = torch.where(eq_mask, ix_eq, ix_po).to(torch.int64)
    iy   = torch.where(eq_mask, iy_eq, iy_po).to(torch.int64)

    # Morton within-face index
    ipf = _morton2D(ix, iy)                     # 0 .. nside*nside - 1 (int64)

    # Global nested id
    nside2 = nside * nside
    pix = face * nside2 + ipf                   # (B,)

    return pix.view(orig_shape)

# ===== helpers =====
def lonlat_to_xyz_torch(lon_deg: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
    lon = torch.deg2rad(lon_deg); lat = torch.deg2rad(lat_deg)
    cl = torch.cos(lat)
    x = cl * torch.cos(lon); y = cl * torch.sin(lon); z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)

def healpix_id_from_lonlat(lon_deg: float, lat_deg: float, nside: int, device="cpu") -> int:
    lon = torch.tensor([lon_deg], device=device)
    lat = torch.tensor([lat_deg], device=device)
    xyz = lonlat_to_xyz_torch(lon, lat)                # (1,3)
    pid = healpix_vec2pix_nest_batch(xyz, nside)[0].item()
    return pid

# ===== int -> RGB mappers (no colormap dependency) =====
def _splitmix64(u: np.ndarray) -> np.ndarray:
    '''SplitMix64 mixer on uint64 array -> uint64 array.'''
    x = u.astype(np.uint64, copy=False)
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(30)); x *= np.uint64(0xBF58476D1CE4E5B9); x &= np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(27)); x *= np.uint64(0x94D049BB133111EB); x &= np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(31))
    return x

def rgb_uint24(ids: np.ndarray, seed: int = 0) -> np.ndarray:
    '''
    Deterministic colors from integer IDs:
      - cast ids -> uint64
      - hash with SplitMix64 (seedable)
      - take top 24 bits as R,G,B
    Returns float RGB in [0,1], shape (...,3).
    '''
    arr_u = np.asarray(ids, dtype=np.uint64)           # ensure unsigned
    h     = _splitmix64(arr_u ^ np.uint64(seed))       # XOR in uint64

    # take the top 24 bits
    rgb24 = (h >> np.uint64(40)).astype(np.uint32)     # keep 24 bits
    r = ((rgb24 >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb24 >>  8) & 0xFF).astype(np.float32) / 255.0
    b = ((rgb24 >>  0) & 0xFF).astype(np.float32) / 255.0
    rgb = np.stack([r, g, b], axis=-1).reshape(ids.shape + (3,))

    # lift very dark colors a bit (optional)
    lum = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]
    rgb = np.where(lum[...,None] < 0.15, np.clip(rgb + 0.15, 0, 1), rgb)
    return rgb

def rgb_hsv_hash(ids: np.ndarray, sat: float = 0.9, val: float = 0.95, seed: int = 0) -> np.ndarray:
    '''
    Map id -> hue via SplitMix64; fixed S,V. Returns float RGB in [0,1], shape (...,3).
    '''
    arr_u = np.asarray(ids, dtype=np.uint64)
    h     = _splitmix64(arr_u ^ np.uint64(seed))
    hue   = (h.astype(np.float64) / float(2**64))  # [0,1)

    H = (hue * 6.0) % 6.0
    C = val * sat
    X = C * (1 - np.abs((H % 2) - 1))

    r = np.where((0<=H)&(H<1), C, np.where((1<=H)&(H<2), X, np.where((2<=H)&(H<3), 0, np.where((3<=H)&(H<4), 0, np.where((4<=H)&(H<5), X, C)))))
    g = np.where((0<=H)&(H<1), X, np.where((1<=H)&(H<2), C, np.where((2<=H)&(H<3), C, np.where((3<=H)&(H<4), X, np.where((4<=H)&(H<5), 0, 0)))))
    b = np.where((0<=H)&(H<1), 0, np.where((1<=H)&(H<2), 0, np.where((2<=H)&(H<3), X, np.where((3<=H)&(H<4), C, np.where((4<=H)&(H<5), C, X)))))

    m = (val - C)
    rgb = np.stack([r+m, g+m, b+m], axis=-1).reshape(ids.shape + (3,))
    return rgb


def id_to_rgb24(ids: np.ndarray) -> np.ndarray:
    '''Pack uint ids into 24-bit RGB (uint8). Invertible.'''
    ids = np.asarray(ids, dtype=np.uint32)
    r = (ids >> 16) & 0xFF
    g = (ids >> 8)  & 0xFF
    b = (ids >> 0)  & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

def rgb24_to_id(rgb: np.ndarray) -> np.ndarray:
    '''Unpack 24-bit RGB back to uint ids.'''
    rgb = np.asarray(rgb, dtype=np.uint32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return ((r << 16) | (g << 8) | b).astype(np.uint32)

# ===== robust visualizer (min–max normalize; safe colormap) =====
# ===== local viewer: window around a lon/lat, colored by id->RGB =====
def plot_healpix_local_rgb(nside: int,
                           center_lon: float, center_lat: float,
                           win_deg: float = 4.0,
                           res: tuple[int,int] = (800, 600),
                           device: str = "cpu",
                           color_mode: str = "hash24",  # "hash24" or "hsv"
                           seed: int = 0,
                           show_borders: bool = True,
                           title: str | None = None,
                           figsize=(8,6), dpi=120,
                           save_path: str | None = None):
    '''
    Visualize a local lon/lat window, coloring tiles by id directly (no colormap).
    Also prints the tile id at the center point.
    '''
    W, H = res
    # window bounds
    lon_min = center_lon - win_deg/2; lon_max = center_lon + win_deg/2
    lat_min = center_lat - win_deg/2; lat_max = center_lat + win_deg/2

    # grid (H,W)
    lon = torch.linspace(lon_min, lon_max, W, device=device)
    lat = torch.linspace(lat_min, lat_max, H, device=device)
    LAT, LON = torch.meshgrid(lat, lon, indexing="ij")  # (H,W)

    # xyz -> ids
    xyz = lonlat_to_xyz_torch(LON.reshape(-1), LAT.reshape(-1))
    pix = healpix_vec2pix_nest_batch(xyz, nside).reshape(H, W)
    ids = pix.cpu().numpy().astype(np.int64)

    # center id
    center_id = healpix_id_from_lonlat(center_lon, center_lat, nside, device=device)
    print(f"[local] nside={nside}, center=({center_lon:.3f},{center_lat:.3f}) ⇒ tile_id={center_id}")

    # id -> RGB
    if color_mode.lower() == "inv":
        rgb = id_to_rgb24(ids)
    elif color_mode.lower() == "hsv":
        rgb = rgb_hsv_hash(ids, sat=0.9, val=0.95, seed=seed)  # (H,W,3)
    else:
        rgb = rgb_uint24(ids, seed=seed)                       # (H,W,3)

    # plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(rgb, extent=(lon_min, lon_max, lat_min, lat_max),
              origin="lower", interpolation="nearest")
    # marker at center + annotate id
    ax.scatter([center_lon], [center_lat], s=16, c="white", edgecolors="black", zorder=5)
    ax.text(center_lon, center_lat, f"  id={center_id}", color="black",
            bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="none"),
            va="center", ha="left", fontsize=9, zorder=6)

    # optional country borders (if available)
    if show_borders:
        try:
            import geopandas as gpd
            from cartopy.io import shapereader as shp
            path = shp.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
            world = gpd.read_file(path)
            try:
                if world.crs is None or world.crs.to_epsg() != 4326:
                    world = world.to_crs(4326)
            except Exception:
                pass
            world.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6, alpha=0.9, zorder=4)
        except Exception as e:
            print(f"[local] Borders overlay skipped: {e}")

    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    if title is None:
        title = f"HEALPix local (nside={nside}) — center id {center_id}"
    ax.set_title(title)
    ax.grid(True, color="k", alpha=0.15, linestyle=":", linewidth=0.5)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.show()
    return center_id, (fig, ax)

def plot_healpix_tiles_safe(nside: int,
                            res: tuple[int,int]=(2048, 1024),
                            device: str = "cpu",
                            cmap: str = "nipy_spectral",
                            show_borders: bool = True,
                            draw_edges: bool = False,
                            title: str | None = None,
                            figsize=(12,6),
                            dpi=120,
                            save_path: str | None = None,
                            borders_color: str = "black",
                            borders_lw: float = 0.7,
                            borders_alpha: float = 0.95):
    '''
    Rasterizes HEALPix (NESTED) IDs to an equirectangular image with a vivid colormap.
    Uses min–max normalization to ensure visible variation (no all-black accidents).
    '''
    W, H = res

    # 1) Build grid with natural image layout: (H, W)
    lat = torch.linspace(-90.0,  90.0,  H, device=device)
    lon = torch.linspace(-180.0, 180.0, W, device=device)
    LAT, LON = torch.meshgrid(lat, lon, indexing="ij")   # (H, W)

    # 2) xyz and tile ids
    xyz = lonlat_to_xyz_torch(LON.reshape(-1), LAT.reshape(-1))  # (H*W, 3)
    pix = healpix_vec2pix_nest_batch(xyz, nside).reshape(H, W)   # (H, W)
    ids = pix.cpu().numpy().astype(np.int64)
    #ids = ids % 256
    
    # 3) Normalize to [0,1] (guarantees full colormap usage)
    vmin = ids.min(); vmax = ids.max()
    norm_img = (ids - vmin) / (max(1, vmax - vmin))  # (H, W) float32-ish

    # 4) Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("white")
    if cmap is None:
        rgb = id_to_rgb24(ids)
        im = ax.imshow(rgb,
                    extent=(-180, 180, -90, 90),
                    origin="lower",
                    interpolation="nearest") 
    else:
        im = ax.imshow(norm_img,
                    extent=(-180, 180, -90, 90),
                    origin="lower",
                    interpolation="nearest",
                    cmap=cmap, vmin=0.0, vmax=1.0,
                    zorder=0)
    

    if draw_edges:
        edges = (ids != np.roll(ids, -1, axis=1)) | (ids != np.roll(ids, -1, axis=0))
        edges[-1, :] = True; edges[:, -1] = True
        edge_img = np.where(edges, 0.0, np.nan).astype(np.float32)
        ax.imshow(edge_img, extent=(-180, 180, -90, 90), origin="lower",
                  cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", zorder=5)

    if show_borders:
        try:
            import geopandas as gpd 
            world_path = _resolve_world_layer(True)   
            world = gpd.read_file(world_path)
            # ensure lon/lat CRS; boundary geometry only
            if world.crs is None or world.crs.to_epsg() != 4326:
                try:
                    world = world.set_crs(4326, allow_override=True)
                except Exception:
                    world = world.to_crs(4326)
            world = world.dropna(subset=["geometry"])
            world.boundary.plot(
                ax=ax,
                edgecolor=borders_color,
                linewidth=borders_lw,
                alpha=borders_alpha,
                zorder=10,
            )
        except Exception as e:
            print(f"[plot_healpix_tiles_safe] Borders overlay skipped: {e}")

    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_aspect("auto")
    ax.grid(True, color="k", alpha=0.15, linestyle=":", linewidth=0.5)
    if title is None:
        title = f"HEALPix (NESTED) — nside={nside}, res={res[0]}×{res[1]}"
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.show()
    return fig, ax

#plot_healpix_tiles_safe(nside=64, res=(2048*2,1024*2), 
                        #cmap='Paired',
#                        cmap=None,
#                        show_borders=True, draw_edges=True)


# Luxembourg approx: lon ~ 6.13° E, lat ~ 49.82° N
#center_lon, center_lat = 6.13, 49.82
#nside = 64  # ~100 km tiles

# See the tile id and its neighborhood, with programmatic int→RGB colors
#plot_healpix_local_rgb(nside, center_lon, center_lat, win_deg=4.0, color_mode="inv")
# or a vivid hue mapping:
#plot_healpix_local_rgb(nside, center_lon, center_lat, win_deg=4.0, color_mode="hsv")


