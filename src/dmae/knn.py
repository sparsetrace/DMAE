import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import csr_matrix

def _symmetrize_union_min_d2(n: int, idx: np.ndarray, d2: np.ndarray) -> csr_matrix:
    """
    Given directed kNN lists (idx,d2) with shape (n,k), build a symmetric CSR using union support.
    If both directions exist, keep min(d2).
    """
    n, k = idx.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = idx.reshape(-1).astype(np.int64)
    vals = d2.reshape(-1).astype(np.float32)

    m = cols >= 0
    rows, cols, vals = rows[m], cols[m], vals[m]

    # Canonical undirected edge key (u < v)
    u = np.minimum(rows, cols)
    v = np.maximum(rows, cols)

    # Drop diagonal just in case
    keep = u != v
    u, v, vals = u[keep], v[keep], vals[keep]

    key = u * np.int64(n) + v
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    u = u[order]
    v = v[order]
    vals = vals[order]

    # Reduce duplicates by min on each (u,v)
    # Find group boundaries
    boundaries = np.empty_like(key, dtype=bool)
    boundaries[0] = True
    boundaries[1:] = key[1:] != key[:-1]
    starts = np.flatnonzero(boundaries)

    # min-reduce by group
    vals_min = np.minimum.reduceat(vals, starts)
    u_uniq = u[starts]
    v_uniq = v[starts]

    # Expand to both directions for symmetric CSR
    rr = np.concatenate([u_uniq, v_uniq]).astype(np.int64)
    cc = np.concatenate([v_uniq, u_uniq]).astype(np.int64)
    dd = np.concatenate([vals_min, vals_min]).astype(np.float32)

    # Build CSR (no duplicates expected after reduce)
    return csr_matrix((dd, (rr, cc)), shape=(n, n), dtype=np.float32)


@jax.jit
def _merge_topk(best_d2, best_idx, cand_d2, cand_idx, k: int):
    """Merge (best) and (cand) per row keeping smallest k by d2."""
    d2_all = jnp.concatenate([best_d2, cand_d2], axis=1)
    idx_all = jnp.concatenate([best_idx, cand_idx], axis=1)

    # lax.top_k returns largest, so use -d2 to get smallest d2
    neg = -d2_all
    vals, pos = jax.lax.top_k(neg, k)
    new_d2 = -vals
    new_idx = jnp.take_along_axis(idx_all, pos, axis=1)
    return new_d2, new_idx


def knn_jax_to_csr(
    R_iX: np.ndarray,
    R_jX: np.ndarray | None = None,
    k: int = 10,
    q_block: int = 1024,
    r_block: int = 8192,
) -> csr_matrix:
    """
    Exact brute-force kNN using JAX (GPU-friendly blocking) returning a SciPy CSR matrix of squared distances.

    - If R_jX is provided: returns directed CSR (n_i, n_j) with exactly k entries per row.
    - If R_jX is None: uses R_iX as reference, excludes self, and returns symmetric union CSR (n, n).

    CSR.data stores squared distances (float32).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if r_block < k:
        raise ValueError("r_block must be >= k because we top-k within each reference block.")

    Xq = np.asarray(R_iX, dtype=np.float32)
    self_case = R_jX is None
    Xr = Xq if self_case else np.asarray(R_jX, dtype=np.float32)

    n_q, d = Xq.shape
    n_r, d2_ = Xr.shape
    if d2_ != d:
        raise ValueError("R_iX and R_jX must have the same feature dimension")

    if self_case:
        if k >= n_r:
            raise ValueError("For self kNN, require k < N (self excluded).")
    else:
        if k > n_r:
            k = n_r  # clamp for query-vs-reference mode

    # Move references to device once
    Xr_dev = jnp.asarray(Xr)
    rnorm_dev = jnp.sum(Xr_dev * Xr_dev, axis=1)  # (n_r,)

    # Output buffers on host
    out_idx = np.empty((n_q, k), dtype=np.int32)
    out_d2 = np.empty((n_q, k), dtype=np.float32)

    # Reference block starts on host (small array)
    r_starts = np.arange(0, n_r, r_block, dtype=np.int32)

    # Jitted kernel for one query block, scanning over fixed r_starts in Python.
    # (This avoids needing to pad to a constant scan length; only last ref-block is smaller.)
    @jax.jit
    def knn_one_qblock(Q_dev, q_ids_dev):
        qnorm = jnp.sum(Q_dev * Q_dev, axis=1)  # (bq,)
        bq = Q_dev.shape[0]

        best_d2 = jnp.full((bq, k), jnp.inf, dtype=jnp.float32)
        best_idx = jnp.full((bq, k), -1, dtype=jnp.int32)

        def body(carry, r_start):
            best_d2, best_idx = carry
            r_start = r_start.astype(jnp.int32)

            # Slice a ref block (may be smaller at end)
            Rb = jax.lax.dynamic_slice(
                Xr_dev,
                (r_start, 0),
                (jnp.minimum(r_block, n_r - r_start), d),
            )
            rnorm_b = jax.lax.dynamic_slice(
                rnorm_dev,
                (r_start,),
                (jnp.minimum(r_block, n_r - r_start),),
            )

            # Compute D2 block
            G = Q_dev @ Rb.T
            d2_block = qnorm[:, None] + rnorm_b[None, :] - 2.0 * G
            d2_block = jnp.maximum(d2_block, 0.0)

            # Exclude self if self_case and overlapping ids
            if self_case:
                r_ids = jnp.arange(r_start, r_start + Rb.shape[0], dtype=jnp.int32)
                mask = (q_ids_dev[:, None] == r_ids[None, :])
                d2_block = jnp.where(mask, jnp.inf, d2_block)

            # Top-k within this block
            neg = -d2_block
            vals, loc = jax.lax.top_k(neg, k)  # requires block width >= k; ensured by r_block>=k and n_r>k
            cand_d2 = -vals
            cand_idx = (r_start + loc).astype(jnp.int32)

            best_d2, best_idx = _merge_topk(best_d2, best_idx, cand_d2, cand_idx, k)
            return (best_d2, best_idx), None

        (best_d2, best_idx), _ = jax.lax.scan(body, (best_d2, best_idx), jnp.asarray(r_starts))
        return best_idx, best_d2

    # Process query blocks on host
    for qs in range(0, n_q, q_block):
        qe = min(qs + q_block, n_q)
        Q = jnp.asarray(Xq[qs:qe])
        q_ids = jnp.arange(qs, qe, dtype=jnp.int32)  # global ids for diagonal masking in self case
        idx_dev, d2_dev = knn_one_qblock(Q, q_ids)
        idx_blk, d2_blk = jax.device_get((idx_dev, d2_dev))
        out_idx[qs:qe] = idx_blk
        out_d2[qs:qe] = d2_blk

    # Build CSR on CPU
    if not self_case:
        indptr = np.arange(0, n_q * k + 1, k, dtype=np.int64)
        return csr_matrix((out_d2.reshape(-1), out_idx.reshape(-1), indptr),
                          shape=(n_q, n_r), dtype=np.float32)

    # Symmetric union CSR (min distance on duplicates)
    return _symmetrize_union_min_d2(n_q, out_idx, out_d2)
