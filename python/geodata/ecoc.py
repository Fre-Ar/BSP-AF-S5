import argparse, json, hashlib
from collections import defaultdict
import numpy as np

# -----------------------------
# Utilities
# -----------------------------

def load_graph(path):
    """Load adjacency dict (keys may be strings), make undirected, deduplicate."""
    with open(path, "r") as f:
        raw = json.load(f)

    # Normalize ids (keep original ints)
    ids = []
    for k in raw.keys():
        try:
            ids.append(int(k))
        except:
            ids.append(k)
    ids = sorted(ids, key=lambda x: (isinstance(x, str), x))

    # Map id -> idx and back
    id2idx = {cid: i for i, cid in enumerate(ids)}
    idx2id = {i: cid for cid, i in id2idx.items()}

    # Build undirected edge list (i<j)
    edges = set()
    for k, nbrs in raw.items():
        try:
            a = int(k)
        except:
            a = k
        ia = id2idx[a]
        for b in nbrs:
            try:
                b = int(b)
            except:
                pass
            if b not in id2idx:  # skip unknown ids (shouldn't happen)
                continue
            ib = id2idx[b]
            if ia == ib:
                continue
            i, j = (ia, ib) if ia < ib else (ib, ia)
            edges.add((i, j))
    edges = sorted(edges)
    return ids, id2idx, idx2id, np.array(edges, dtype=np.int32)

def bit_hash(i, k, salt=b"BSPv1"):
    """Deterministic 64-bit hash for (salt, bit_index, class_id)."""
    if isinstance(i, int):
        i_bytes = i.to_bytes(4, "little", signed=False)
    else:
        i_bytes = str(i).encode("utf-8")
    h = hashlib.blake2b(salt + k.to_bytes(2, "little") + i_bytes, digest_size=8).digest()
    return int.from_bytes(h, "little")

def init_balanced_codes(ids, K=32, salt=b"BSPv1"):
    """
    Construct an NxK 0/1 matrix with (almost) exactly half ones per column,
    by sorting per-column hash scores and setting top half to 1.
    Deterministic and well spread.
    """
    N = len(ids)
    codes = np.zeros((N, K), dtype=np.uint8)
    for k in range(K):
        scores = np.array([bit_hash(ids[i], k, salt) for i in range(N)], dtype=np.uint64)
        # Top half -> 1, bottom half -> 0
        order = np.argsort(scores)
        half = N // 2
        ones_idx = order[-half:]
        codes[ones_idx, k] = 1
        # If N is odd, allow imbalance of at most 1
        # (columns will differ by <=1 between ones/zeros)
    return codes

def pack_row_bits_to_uint32(row_bits):
    """Pack 32 bits (least-significant bit is col 0) into a Python int (fits JSON safely)."""
    # Ensure length 32
    assert row_bits.shape[0] <= 32
    # Pack little-endian bit order
    val = 0
    for b in range(row_bits.shape[0]):
        if row_bits[b]:
            val |= (1 << b)
    return int(val)

def hamming_neighbors(codes, edges):
    """Vectorized Hamming distances ONLY over neighbor edge list."""
    # codes: [N,K] uint8 in {0,1}; edges: [E,2]
    a = codes[edges[:, 0]]  # [E,K]
    b = codes[edges[:, 1]]  # [E,K]
    return (a ^ b).sum(axis=1)  # [E]

# -----------------------------
# Refinement (neighbor maximin)
# -----------------------------

def refine_codes_neighbor_maximin(codes, edges, steps=20000, balance_tol=1, seed=42):
    """
    Greedy lexicographic refinement:
      maximize (min_neighbor_distance, mean_neighbor_distance)
    by flipping single bits while keeping each column within balance tolerance.
    Efficiently updates neighbor distances affected by a flip.

    Deterministic iteration order given seed.
    """
    rng = np.random.default_rng(seed)
    N, K = codes.shape
    E = edges.shape[0]

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
    d = hamming_neighbors(codes, edges)  # [E]
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
            # (skip or allow to diversify globally—here we skip)
            continue

        # For each affected edge (i, j), distance changes by ±1 depending on bit k of j
        # prev_bit_diff = codes[i,k] XOR codes[j,k]
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
# Main
# -----------------------------

def main():
    LAYER = "countries"
    ID = "id"
    PATH = "python/geodata/geodata_adjacency.json"
    OUT = "python/geodata/countries.ecoc.json"
    
    BITS = 32
    BALANCE_TOL = 1
    STEPS = 20_000
    SALT = "BSPv1" 
    SEED = 42

    # Load graph
    ids, id2idx, idx2id, edges = load_graph(PATH)
    N = len(ids)
    K = BITS
    print(f"Loaded {N} nodes and {edges.shape[0]} undirected edges.")

    # Initial balanced codes (deterministic)
    codes = init_balanced_codes(ids, K=K, salt=SALT.encode("utf-8"))

    # Optional: if K<32, we still pack into 32 by leaving higher bits zero.
    if K < 32:
        pad = np.zeros((N, 32 - K), dtype=np.uint8)
        codes = np.concatenate([codes, pad], axis=1)

    # Neighbor-aware refinement
    codes, dmin, dmean = refine_codes_neighbor_maximin(
        codes, edges, steps=STEPS, balance_tol=BALANCE_TOL, seed=SEED
    )
    print(f"Neighbor Hamming distance: min={dmin}, mean={dmean:.2f}")

    # Build id -> uint32 dict
    out = {}
    for i in range(N):
        cid = idx2id[i]
        val = pack_row_bits_to_uint32(codes[i, :32])
        # JSON keys must be strings if you want to preserve large ints cross-language;
        # here values fit in 32 bits so numeric is safe. We'll keep keys as strings to be safe.
        out[str(cid)] = val

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    # Also dump some quick stats (optional)
    stats_path = OUT.replace(".json", ".stats.json")
    # Compute global pairwise mean (diagnostic only)
    # (Upper triangle distances; small N so okay)
    # Note: if this is slow for you, you can skip it.
    diffs = []
    for i in range(N):
        x = codes[i:i+1, :32]
        y = codes[i+1:, :32]
        if y.shape[0] == 0: break
        d = (x ^ y).sum(axis=1)
        diffs.append(d)
    if diffs:
        allpair_mean = float(np.concatenate(diffs).mean())
    else:
        allpair_mean = 0.0
    stats = {"N": N, "K": int(K), "neighbor_min": int(dmin), "neighbor_mean": dmean, "allpair_mean": allpair_mean}
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote codes to {OUT} and stats to {stats_path}")

if __name__ == "__main__":
    main()
