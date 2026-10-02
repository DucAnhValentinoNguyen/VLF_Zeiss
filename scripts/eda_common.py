"""Shared helpers for the dataset EDA scripts (perceptual-hash redundancy analysis)."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

THRESHOLDS = (0, 2, 6)
NN_BINS = 33  # nearest-neighbour Hamming distance histogram: 0..31, then >=32


def redundancy(hashes: np.ndarray) -> dict:
    """Redundancy statistics for a set of 64-bit perceptual hashes.

    Images with identical hashes form a group; groups are linked when their hashes are
    within t bits. For each threshold t:
      with_neighbor_le{t}: images with at least one OTHER image within t bits
      clusters_le{t}:      connected components (single linkage) over images
      redundant_le{t}:     n - clusters, the images a de-duplication at t would remove
    plus nn_hist: histogram of each image's nearest-neighbour Hamming distance
    (0 when its hash is repeated).
    """
    hashes = np.asarray(hashes, dtype=np.uint64)
    n = int(hashes.size)
    out = {"n": n}
    if n == 0:
        return out
    uniq, inv, cnt = np.unique(hashes, return_inverse=True, return_counts=True)
    u = uniq.size
    nn = np.full(u, 64, dtype=np.int64)
    edges = {t: ([], []) for t in THRESHOLDS if t > 0}
    chunk = 256
    for i in range(0, u, chunk):
        d = np.bitwise_count(np.bitwise_xor(uniq[i:i + chunk, None], uniq[None, :])).astype(np.int16)
        r = np.arange(d.shape[0])
        d[r, r + i] = 99  # ignore self
        nn[i:i + chunk] = d.min(axis=1)
        for t, (a, b) in edges.items():
            ri, ci = np.nonzero(d <= t)
            a.append(ri + i)
            b.append(ci)
    nn_img = np.where(cnt > 1, 0, nn)
    hist = np.zeros(NN_BINS, dtype=np.int64)
    np.add.at(hist, np.minimum(nn_img, NN_BINS - 1), cnt)
    out["nn_hist"] = hist.tolist()
    for t in THRESHOLDS:
        if t == 0:
            n_clusters = u
        else:
            a = np.concatenate(edges[t][0]) if edges[t][0] else np.zeros(0, dtype=np.int64)
            b = np.concatenate(edges[t][1]) if edges[t][1] else np.zeros(0, dtype=np.int64)
            g = coo_matrix((np.ones(a.size, dtype=np.int8), (a, b)), shape=(u, u))
            n_clusters, _ = connected_components(g, directed=False)
        out[f"clusters_le{t}"] = int(n_clusters)
        out[f"redundant_le{t}"] = int(n - n_clusters)
        out[f"with_neighbor_le{t}"] = int(cnt[(cnt > 1) | (nn <= t)].sum())
    return out
