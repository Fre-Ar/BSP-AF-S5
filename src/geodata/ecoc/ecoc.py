# src/geodata/ecoc/ecoc.py

from typing import Dict, Literal
from pathlib import Path
import json, hashlib
import pyarrow.parquet as pq
import numpy as np
import torch
import torch.nn.functional as F

from .adjacency import load_graph
from utils.utils_geo import SEED

# -----------------------------
# CONFIG
# -----------------------------

# Salt for hashing class ids into bits. Changing this changes all codes.
SALT = b"BSPv1"
# Default refinement budget (# of proposed bit flips).
STEPS = 20_000
# Allowed imbalance per column around N//2 ones (in absolute count).
BALANCE_TOL = 1
# Default bit-length for ECOC codes
BIT_LENGTH = 32

# -----------------------------
# Utilities
# -----------------------------

def _bit_hash(i, k: int, salt: bytes = SALT):
    '''
    Deterministic 64-bit hash for (salt, bit_index, class_id).
    
    Parameters
    ----------
    i : int or str
        Class identifier. Integer ids are encoded as fixed 4-byte little-endian;
        non-integer ids are encoded as UTF-8 strings.
    k : int
        Bit index in the ECOC code (0-based).
    salt : bytes, optional
        Extra salt to inject into the hash.

    Returns
    -------
    int
        A 64-bit integer obtained by hashing (salt, k, i) with BLAKE2b.
    '''
    # encode id into bytes
    if isinstance(i, int):
        i_bytes = i.to_bytes(4, "little", signed=False)
    else:
        i_bytes = str(i).encode("utf-8")
    
    # hash the encoded id to produce reproducible pseudorandom score per (id, bit)
    k_bytes = k.to_bytes(2, "little", signed=False)
    h = hashlib.blake2b(salt + k_bytes + i_bytes, digest_size=8).digest()
    
    # decode int from hashed bytes
    return int.from_bytes(h, "little")

def _init_balanced_codes(ids: list, K: int = BIT_LENGTH, salt: bytes = SALT):
    '''
    Initializes an NxK 0/1 matrix with (almost) exactly half ones per column.
    
    Parameters
    ----------
    ids : list
        List of class ids (int or str). The ordering of ids is preserved. 
        N is its length (number of classes).
    K : int
        Number of bits (columns) in the ECOC code.
    salt : bytes, optional
        Salt passed to `bit_hash`, controlling the hash-derived scores.

    Returns
    -------
    codes : ndarray
        Array of shape (N, K), where each column has roughly half
        of its entries equal to 1 and half equal to 0 (imbalance at most 1
        when N is odd).
    '''
    N = len(ids)
    codes = np.zeros((N, K), dtype=np.uint8)
    for k in range(K):
        # Compute a hash score for every id
        scores = np.array([_bit_hash(ids[i], k, salt) for i in range(N)], dtype=np.uint64)
        
        # Sort ids by the hash score 
        order = np.argsort(scores)
        # and assign the top half to 1, bottom half to 0.
        half = N // 2
        ones_idx = order[-half:]
        codes[ones_idx, k] = 1
        
        # If N is odd, allow imbalance of at most 1
        # (columns will differ by <=1 between ones/zeros)
    return codes

def _pack_row_bits_to_int(row_bits: np.ndarray):
    '''
    Packs an arbitrary-length 0/1 bit row into a Python int.
    Bit i is stored in column i (Least Significant Bit first).
    
    Parameters
    ----------
    row_bits : ndarray
        1D array of shape (K,) with values in {0, 1}, representing one code row.

    Returns
    -------
    int
        Integer whose binary representation encodes the bits of `row_bits`:
    '''
    val = 0
    for b in range(row_bits.shape[0]):
        if row_bits[b]:
            val |= (1 << b)
    return int(val)

def _code_to_bits_np(code: int, n_bits: int = BIT_LENGTH) -> np.ndarray:
    """
    Converts an integer code into a {0,1} bit vector (LSB-first).

    Parameters
    ----------
    code : int
        Integer-encoded ECOC (e.g. 32-bit).
    n_bits : int, optional
        Number of bits to keep (defaults to BIT_LENGTH).

    Returns
    -------
    bits : np.ndarray
        Array of shape (n_bits,) with entries in {0,1}, where
        bit 0 corresponds to column 0 (least-significant bit).
    """
    return np.array([(code >> b) & 1 for b in range(n_bits)], dtype=np.uint8)



def _hamming_neighbors(codes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    '''
    Computes Hamming distances only across neighbor edges.
    
    Parameters
    ----------
    codes : ndarray
        Array of shape (N, K), dtype uint8, with entries in {0, 1}.
    edges : ndarray
        Array of shape (E, 2), dtype int32 or int64, where each row (i, j)
        is an undirected edge between nodes i and j.

    Returns
    -------
    distances : ndarray
        Array of shape (E,), dtype int32, where `distances[e]` is the Hamming
        distance between `codes[edges[e,0]]` and `codes[edges[e,1]]`.
    '''
    # codes: [N,K] uint8 in {0,1}; edges: [E,2]
    a = codes[edges[:, 0]]  # [E,K]
    b = codes[edges[:, 1]]  # [E,K]
    return (a ^ b).sum(axis=1)  # [E]

# -----------------------------
# Refinement (neighbor maximin)
# -----------------------------

def _refine_codes_neighbor_maximin(codes: np.ndarray,
                                  edges: np.ndarray,
                                  seed: int = SEED,
                                  steps=STEPS,
                                  balance_tol=BALANCE_TOL):
    '''
    Refines ECOC codes to maximize neighbor Hamming distances (maximin objective) 
    by flipping single bits while keeping each column within balance tolerance.
    
    Parameters
    ----------
    codes : ndarray
        Initial codes of shape (N, K), with values in {0, 1}.
    edges : ndarray
        Edge list of shape (E, 2). Each row (i, j) is
        an undirected edge between node indices i and j.
    seed : int
        Random seed controlling the proposal order of (i, k) bit flips.
    steps : int, optional
        Number of flip proposals to consider.
    balance_tol : int, optional
        Allowed deviation from perfect column balance. Each column's number
        of ones must remain within `target ± balance_tol`, where `target = N//2`.

    Returns
    -------
    codes : ndarray
        Refined codes (same shape as input), potentially with some bits flipped.
    best_min : int
        Final minimum neighbor Hamming distance across all edges.
    best_mean : float
        Final mean neighbor Hamming distance across all edges.
    '''
    rng = np.random.default_rng(seed)
    N, K = codes.shape

    # Precompute: for each node, which edge indices touch it?
    touch = [[] for _ in range(N)]
    for e_idx, (i, j) in enumerate(edges):
        touch[i].append(e_idx)
        touch[j].append(e_idx)
    touch = [np.array(lst, dtype=np.int32) for lst in touch]

    # Column sums to enforce balance
    col_sums = codes.sum(axis=0).astype(np.int32)
    target = N // 2  # desired ones per column (±1 if N odd)

    # Current neighbor distances
    d = _hamming_neighbors(codes, edges)  # [E]
    best_min = int(d.min())
    best_mean = float(d.mean())

    # Deterministic order of proposals but shuffled once:
    all_pairs = [(i, k) for i in range(N) for k in range(K)]
    rng.shuffle(all_pairs)

    for t in range(steps):
        i, k = all_pairs[t % len(all_pairs)]

        # Enforce column balance (stay within ±balance_tol of N//2)
        current = codes[i, k]
        new_val = current ^ 1
        delta_col = 1 if new_val == 1 else -1
        if abs((col_sums[k] + delta_col) - target) > balance_tol:
            continue  # would unbalance this column too much

        # Compute effect on neighbor distances for edges touching i
        idxs = touch[i]
        if idxs.size == 0:
            # isolated node; flipping does not affect neighbor objective
            continue

        # For each affected edge (i, j), distance changes by ±1 depending on bit k of j
        partners = np.where(edges[idxs, 0] == i, edges[idxs, 1], edges[idxs, 0])
        prev_diff_bit = (codes[partners, k] ^ current).astype(np.int8)  # 0 or 1
        delta = (1 - 2 * prev_diff_bit).astype(np.int16)  # +1 if prev 0, -1 if prev 1

        old_vals = d[idxs].copy()
        d[idxs] = old_vals + delta  # tentative update

        new_min = int(d.min())
        new_mean = float(d.mean())

        # Lexicographic accept: improve min, or equal min and improve mean
        accept = (new_min > best_min) or (new_min == best_min and new_mean > best_mean)

        if accept:
            # Commit flip
            codes[i, k] = new_val
            col_sums[k] += delta_col
            best_min, best_mean = new_min, new_mean
        else:
            # Revert neighbor distances
            d[idxs] = old_vals

    return codes, best_min, best_mean

# -----------------------------
# Generation 
# -----------------------------

def gen_ecoc(path,
             out_path,
             bits = BIT_LENGTH,
             seed = SEED,
             balance_tol=BALANCE_TOL,
             steps=STEPS,
             salt=SALT):
    """
    Generates ECOC codes for a graph-defined adjacency and write thems to JSON.
    Also writes a companion stats file with the minimum and mean hamming distances given by this ecoc.

    Parameters
    ----------
    path : str
        Path to an adjacency JSON file of the form
        ``{ node_id: [neighbor_id1, neighbor_id2, ...], ... }``.
        Node ids and neighbor ids may be strings or integers.
    out_path : str
        Path to the output JSON file where ECOC codes will be written.
        The file will map stringified ids to packed integer codes.
    bits : int
        Number of bits per class in the ECOC code. Must satisfy
        ``2**bits > N`` where N is the number of unique node ids, so that
        there are enough distinct codewords.
    seed : int
        Random seed controlling the order of candidate bit flips during refinement.
    balance_tol : int, optional
        Allowed deviation from perfect column balance. Each column must keep its
        number of ones within ``target ± balance_tol``, where ``target = N//2``.
    steps : int, optional
        Maximum number of bit-flip proposals to evaluate in the refinement loop.
    salt : bytes, optional
        Salt passed to `init_balanced_codes` / `bit_hash`, which determines the
        initial hash-based assignment of bits.

    Notes
    -----
    HOW:
    1) Load the graph with `load_graph(path)`, which normalizes node ids and
       returns an undirected edge list over contiguous indices.
    2) Check capacity: require `2**bits > N` so that the code space is strictly
       larger than the number of classes.
    3) Construct an initial N×K code matrix using `init_balanced_codes`, which
       enforces per-column balance via hash-based scores.
    4) Refine codes using `refine_codes_neighbor_maximin`, which greedily
       maximizes the minimum and mean Hamming distances over neighbor edges.
    5) Pack each row of K bits into a Python int (`pack_row_bits_to_int`) and
       write the mapping `id -> int` to JSON.
    6) As a diagnostic, compute the mean Hamming distance over all unordered
       pairs of codes (upper triangle) and write these stats to a separate JSON.
    """

    # Load graph (ids, mappings, and undirected edge list)
    ids, id2idx, idx2id, edges = load_graph(path)
    N = len(ids)
    K = bits
    print(f"Loaded {N} nodes and {edges.shape[0]} undirected edges.")

    # Capacity check: we need at least one distinct code per class.
    # In other words, the code space {0,1}^K must be strictly larger than N.
    if (1 << K) <= N:
        raise ValueError(
            f"bits={K} too small for N={N} unique ids: "
            f"need 2^bits > N, but 2^{K}={1<<K}."
        )

    # Initial balanced codes (deterministic)
    codes = _init_balanced_codes(ids, K=K, salt=salt)

    # Neighbor-aware refinement: try to maximize (min_d, mean_d)
    codes, dmin, dmean = _refine_codes_neighbor_maximin(
        codes, edges, steps=steps, balance_tol=balance_tol, seed=seed
    )
    print(f"Neighbor Hamming distance: min={dmin}, mean={dmean:.2f}")

    # Build id -> int(code_bits) dictionary
    ecoc_dict = {}
    for i in range(N):
        cid = idx2id[i]
        # Pack K bits (column 0 -> LSB, etc.) into a Python integer
        val = _pack_row_bits_to_int(codes[i, :K])
        # Using string keys for safety / cross-language robustness
        ecoc_dict[str(cid)] = val

    with open(out_path, "w") as f:
        json.dump(ecoc_dict, f, indent=2)

    # Also dump some quick stats
    stats_path = out_path.replace(".json", ".stats.json")
    
    # Compute global pairwise mean (upper triangle only).
    diffs = []
    for i in range(N):
        x = codes[i:i+1, :K] # shape [1, K]
        y = codes[i+1:, :K]  # shape [N-i-1, K]
        if y.shape[0] == 0: break
        d = (x ^ y).sum(axis=1)
        diffs.append(d)
    
    if diffs:
        allpair_mean = float(np.concatenate(diffs).mean())
    else:
        allpair_mean = 0.0
    
    stats = {
        "N": N,
        "K": int(K),
        "neighbor_min": int(dmin),
        "neighbor_mean": dmean,
        "allpair_mean": allpair_mean
    }
    
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Wrote codes to {out_path} and stats to {stats_path}")


# -----------------------------
# Loading
# -----------------------------
def load_ecoc_codes(path: str | Path, n_bits: int = BIT_LENGTH) -> dict[int, np.ndarray]:
    """
    Loads ECOC map: class_id (int) -> ndarray shape (n_bits,) with {0,1}.

    Parameters
    ----------
    path : str or Path
        JSON file produced by `gen_ecoc`, mapping str(class_id) -> uint32 code.
    n_bits : int, optional
        Number of bits to decode for each code.

    Returns
    -------
    codebook : dict[int, ndarray]
        Mapping from class id to bit vector in {0,1}.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): _code_to_bits_np(int(v), n_bits=n_bits) for k, v in raw.items()}


# -----------------------------
# ECOC Bit Prevalence Weighting
# -----------------------------

def ecoc_prevalence_by_bit(
    parquet_path: str | Path,
    id_to_bits: dict[int, np.ndarray],
    id_col: str = "c1_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes per-bit prevalence of 1s for an ECOC head given id -> bits mapping.

    Parameters
    ----------
    parquet_path : str or Path
        Path to Parquet file with a column `id_col` of integer class IDs.
    id_to_bits : dict[int, ndarray]
        ECOC codebook mapping class id -> bit vector of shape (n_bits,).
    id_col : str, optional
        Name of the column containing class ids (e.g. "c1_id" or "c2_id").

    Returns
    -------
    ones : ndarray
        (n_bits,) int64, total count of 1s per bit.
    totals : ndarray
        (n_bits,) int64, total number of samples accumulated.
    p_ones : ndarray
        (n_bits,) float64, prevalence in [0,1].
    """
    parquet_path = Path(parquet_path)
    # infer n_bits from first item
    some_bits = next(iter(id_to_bits.values()))
    n_bits = int(some_bits.shape[0])

    # init arrays
    ones   = np.zeros(n_bits, dtype=np.int64)
    totals = np.zeros(n_bits, dtype=np.int64)
    zeros_default = np.zeros(n_bits, dtype=np.int8)  # fallback if an ID is missing

    # read the data
    pf = pq.ParquetFile(str(parquet_path))
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[id_col])
        ids = tbl[id_col].to_numpy(zero_copy_only=False)

        # Map IDs -> bit rows. Row-group sized stack keeps memory bounded.
        # NOTE: if a class ID is missing from dict, we use zeros_default.
        bits_mat = np.stack(
            [id_to_bits.get(int(i), zeros_default) for i in ids],
            axis=0
        ).astype(np.int64)

        ones   += bits_mat.sum(axis=0)
        totals += bits_mat.shape[0]

    # computing the prevalence
    p_ones = ones / np.maximum(totals, 1)
    
    return ones, totals, p_ones
  
def pos_weight_from_prevalence(p_ones: np.ndarray, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute PyTorch BCEWithLogitsLoss pos_weight = N_neg / N_pos per bit.

    Parameters
    ----------
    p_ones : ndarray
        Prevalence of 1s per bit in [0,1].
    eps : float, optional
        Small value to clip probabilities away from 0 and 1.

    Returns
    -------
    pos_weight : torch.Tensor
        1D tensor[K] suitable for BCEWithLogitsLoss(pos_weight=...).
    """
    p = np.clip(p_ones, eps, 1.0 - eps)
    pw = (1.0 - p) / p
    return torch.tensor(pw, dtype=torch.float32)


# -----------------------------
# ECOC Decoding
# -----------------------------

def per_bit_threshold(
    pos_weight: torch.Tensor | None,
    device: torch.device | str,
    n_bits: int
):
    """
    Thresholds that are consistent with BCEWithLogitsLoss(pos_weight).

    If pos_weight is None, all thresholds are 0.5.
    Otherwise, t_k = 1 / (1 + w_k).

    Parameters
    ----------
    pos_weight : torch.Tensor or None
        1D tensor[K] of N_neg / N_pos per bit, or a scalar tensor, or None.
    device : device or str
        Torch device where the threshold tensor should live.
    n_bits : int
        Number of bits (K).

    Returns
    -------
    thr : torch.Tensor
        1D tensor[K] of thresholds in (0,1).
    """
    if pos_weight is None:
        return torch.full((n_bits,), 0.5, device=device)
    
    pw = pos_weight.to(device=device, dtype=torch.float32, non_blocking=True)
    if pw.numel() == 1:
        pw = pw.expand(n_bits)
    return 1.0 / (1.0 + pw) # elementwise

@torch.no_grad()
def _prepare_codebook_tensor(
    codebook: Dict[int, np.ndarray],
    device,
    dtype=torch.float32):
    """
    Converts python dict id -> ndarray bits into a tensor on device.

    Returns
    -------
    keys : list[int]
        Sorted class IDs of shape [C].
    codes : torch.Tensor
        Tensor of shape [C, K] in {0,1}.
    """
    keys = sorted(codebook.keys())
    codes = torch.as_tensor(
        np.stack([codebook[k] for k in keys], axis=0),
        dtype=dtype,
        device=device
    )  # [C, K] in {0,1}
    return keys, codes


@torch.no_grad()
def ecoc_decode(
    bits_logits: torch.Tensor,       # [B, K] pre-sigmoid
    codebook: Dict[int, np.ndarray], # {class_id: np.array([0/1]*K)}
    pos_weight=None,                 # float | 1D tensor[K] | None
    mode: Literal['soft', 'hard'] = 'soft'
):
    """
    ECOC decoding consistent with BCEWithLogitsLoss(pos_weight).

    Implements per-bit logit shift: z'_k = z_k - tau_k, tau_k = -log(pos_weight_k).
    Scores each codeword via summed log-likelihood over bits and returns
    the argmax class id per sample.

    Returns
    -------
    pred_class_ids : LongTensor
        Shape [B], containing predicted class IDs.
    """
    """
    Decodes ECOC predictions into class IDs, in a way consistent with
    `BCEWithLogitsLoss(pos_weight=...)`.

    Parameters
    ----------
    bits_logits : torch.Tensor
        Tensor of shape [B, K] with pre-sigmoid logits for each ECOC bit.
    codebook : dict[int, np.ndarray]
        Mapping from class_id -> bit vector of shape (K,) with values in {0, 1}.
        All entries must have the same length K.
    pos_weight : float | torch.Tensor, optional
        The `pos_weight` used in BCEWithLogitsLoss:
          - If None, bits are treated as balanced (threshold at 0.5 probability).
          - If scalar, the same weight is used for all bits.
          - If 1D tensor[K], a separate weight is used per bit.
        For both modes, this determines per-bit decision boundaries via
        the logit shift tau_k = -log(pos_weight_k).
    mode : {"soft", "hard"}, optional
        Decoding mode:
          - "soft": probability-consistent decoding. We apply the logit shift
            z'_k = z_k - tau_k and score each codeword by summed log-likelihood:
                sum_k [ b_k * logσ(z'_k) + (1-b_k) * logσ(-z'_k) ],
            then return the argmax class.
          - "hard": thresholded bits + nearest neighbour in Hamming space.
            We threshold per bit via (z_k > tau_k) and choose the codeword
            with minimal Hamming distance.

    Returns
    -------
    pred_class_ids : torch.LongTensor
        Tensor of shape [B] with predicted class IDs (same domain as `codebook` keys).

    Notes
    -----
    - With pos_weight = N_neg / N_pos, the implicit probability threshold is
      t_k = 1 / (1 + pos_weight_k), and the corresponding logit threshold is
      tau_k = -log(pos_weight_k).
    """
    if mode not in ("soft", "hard"):
        raise ValueError(f"Invalid mode '{mode}', expected 'soft' or 'hard'.")
    
    device = bits_logits.device
    dtype  = bits_logits.dtype
    B, K   = bits_logits.shape

    # prepare codebook
    keys, codes = _prepare_codebook_tensor(codebook, device, dtype)  # codes: [C,K]
    keys_tensor = torch.as_tensor(keys, device=device, dtype=torch.long)
    C = codes.shape[0]

    # Build per-bit shift tau (so decision boundary is at 0)
    if pos_weight is None:
        tau = 0.0 # balanced classes
    else:
        tau = pos_weight
        # move tau to device
        if not torch.is_tensor(tau):
            tau = torch.tensor(tau, dtype=dtype, device=device)
        else:
            tau = tau.to(dtype=dtype, device=device)
        # broadcast tau to K
        if tau.numel() == 1:
            tau = tau.expand(K)
        # compute tau_k = -log pos_weight_k
        tau = (-torch.log(tau)).view(1, K)  # [1,K]  

    if mode == "soft":
        # log-likelihood-based decoding
        z_adj = bits_logits - tau  # [B, K]

        # Broadcast to classes
        Z = z_adj.unsqueeze(1)          # [B, 1, K]
        BITS = codes.unsqueeze(0)       # [1, C, K]

        # Log-likelihood for each class and sample
        score = BITS * F.logsigmoid(Z) + (1.0 - BITS) * F.logsigmoid(-Z)  # [B, C, K]
        cls_idx = score.sum(dim=-1).argmax(dim=1)                          # [B]

    elif mode == "hard":
        # Hard bit decisions: (z > tau) <=> (sigmoid(z) > 1/(1+w))
        pred_bits = (bits_logits > tau).to(torch.float32)            # [B, K]

        # Hamming distance to each code: sum |pred_bits - code|
        dists = (pred_bits.unsqueeze(1) - codes.unsqueeze(0)).abs().sum(dim=-1)  # [B, C]
        cls_idx = dists.argmin(dim=1)  # [B]
    
    # Map argmax indices back to class IDs
    pred_class_ids = keys_tensor[cls_idx]  # [B]
    return pred_class_ids


def _codebook_to_bits_matrix_local(codebook: Dict[int, np.ndarray], n_bits: int | None = None):
    """
    From {class_id: np.uint8[K]} build:
      ids:  [C] int64 array of class ids (sorted ascending)
      bits: [C,K] float tensor (0/1), where K = n_bits or inferred from entries.
    """
    ids = np.array(sorted(codebook.keys()), dtype=np.int64)
    K_inf = int(next(iter(codebook.values())).shape[0])
    K = int(n_bits if n_bits is not None else K_inf)
    M = np.zeros((len(ids), K), dtype=np.uint8)
    for i, cid in enumerate(ids):
        v = codebook[cid]
        if v.shape[0] < K:
            raise ValueError(f"Code for class {cid} has length {v.shape[0]} < requested {K}")
        M[i, :] = v[:K].astype(np.uint8)
    bits = torch.from_numpy(M.astype(np.float32))
    return ids, bits  # np.int64[C], torch.float32[C,K]

@torch.no_grad()
def _ecoc_decode_soft_old(
    logits_bits: torch.Tensor,   # [B, K] pre-sigmoid
    bits_codebook: torch.Tensor, # [C, K] in {0,1}
    tau: float = 1.0
) -> torch.Tensor:
    """
    Soft ECOC log-likelihood (no pos_weight shifts):
      score(c) = sum_k [ b_ck * log σ(z_k/tau) + (1-b_ck) * log σ(-z_k/tau) ]
    Returns indices in [0..C-1].
    """
    z = logits_bits / max(1e-6, float(tau))     # [B,K]
    B = bits_codebook.float()                   # [C,K]
    Z = z.unsqueeze(1)                          # [B,1,K]
    score = B.unsqueeze(0) * F.logsigmoid(Z) + (1 - B).unsqueeze(0) * F.logsigmoid(-Z)  # [B,C,K]
    return score.sum(dim=-1).argmax(dim=1)      # [B]
