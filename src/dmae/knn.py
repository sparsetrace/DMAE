import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import csr_matrix
from functools import partial


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

    u = np.minimum(rows, cols)
    v = np.maximum(rows, cols)

    keep = u != v
    u, v, vals = u[keep], v[keep], vals[keep]

    key = u * np.int64(n) + v
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    u = u[order]
    v = v[order]
    vals = vals[order]

    boundaries = np.empty_like(key, dtype=bool)
    boundaries[0] = True
    boundaries[1:] = key[1:] != key[:-1]
    starts = np.flatnonzero(boundaries)

    vals_min = np.minimum.reduceat(vals, starts)
    u_uniq = u[starts]
    v_uniq = v[starts]

    rr = np.concatenate([u_uniq, v_uniq]).astype(np.int64)
    cc = np.concatenate([v_uniq, u_uniq]).astype(np.int64)
    dd = np.concatenate([vals_min, vals_min]).astype(np.float32)

    return csr_matrix((dd, (rr, cc)), shape=(n, n), dtype=np.float32)


@partial(jax.jit, static_argnums=(4,))
def _merge_topk(best_d2, best_idx, cand_d2, cand_idx, k: int):
    d2_all = jnp.concatenate([best_d2, cand_d2], axis=1)
    idx_all = jnp.concatenate([best_idx, cand_idx], axis=1)

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
            k = n_r

    # Pad reference set so every dynamic_slice has fixed size r_block
    n_blocks = int(np.ceil(n_r / r_block))
    n_r_pad = n_blocks * r_block
    pad = n_r_pad - n_r

    if pad > 0:
        Xr_pad = np.pad(Xr, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)
    else:
        Xr_pad = Xr

    valid_ref = np.arange(n_r_pad, dtype=np.int32) < n_r

    Xr_dev = jnp.asarray(Xr_pad)
    rnorm_dev = jnp.sum(Xr_dev * Xr_dev, axis=1)          # (n_r_pad,)
    valid_ref_dev = jnp.asarray(valid_ref)                # (n_r_pad,)
    r_starts = np.arange(0, n_r_pad, r_block, dtype=np.int32)

    out_idx = np.empty((n_q, k), dtype=np.int32)
    out_d2 = np.empty((n_q, k), dtype=np.float32)

    @jax.jit
    def knn_one_qblock(Q_dev, q_ids_dev):
        qnorm = jnp.sum(Q_dev * Q_dev, axis=1)  # (bq,)
        bq = Q_dev.shape[0]

        best_d2 = jnp.full((bq, k), jnp.inf, dtype=jnp.float32)
        best_idx = jnp.full((bq, k), -1, dtype=jnp.int32)

        def body(carry, r_start):
            best_d2, best_idx = carry
        
            Rb = jax.lax.dynamic_slice(Xr_dev, (r_start, 0), (r_block, d))
            rnorm_b = jax.lax.dynamic_slice(rnorm_dev, (r_start,), (r_block,))
            valid_b = jax.lax.dynamic_slice(valid_ref_dev, (r_start,), (r_block,))
        
            G = Q_dev @ Rb.T
            d2_block = qnorm[:, None] + rnorm_b[None, :] - 2.0 * G
            d2_block = jnp.maximum(d2_block, 0.0)
        
            # Mask padded references
            d2_block = jnp.where(valid_b[None, :], d2_block, jnp.inf)
        
            # Exclude self for self-kNN
            if self_case:
                r_ids = r_start + jnp.arange(r_block, dtype=jnp.int32)
                mask_self = (q_ids_dev[:, None] == r_ids[None, :])
                d2_block = jnp.where(mask_self, jnp.inf, d2_block)
        
            neg = -d2_block
            vals, loc = jax.lax.top_k(neg, k)
            cand_d2 = -vals
            cand_idx = (r_start + loc).astype(jnp.int32)
        
            cand_valid = cand_idx < n_r
            cand_d2 = jnp.where(cand_valid, cand_d2, jnp.inf)
            cand_idx = jnp.where(cand_valid, cand_idx, -1)
        
            best_d2, best_idx = _merge_topk(best_d2, best_idx, cand_d2, cand_idx, k)
            return (best_d2, best_idx), None

        (best_d2, best_idx), _ = jax.lax.scan(body, (best_d2, best_idx), jnp.asarray(r_starts))
        return best_idx, best_d2

    for qs in range(0, n_q, q_block):
        qe = min(qs + q_block, n_q)
        Q = jnp.asarray(Xq[qs:qe])
        q_ids = jnp.arange(qs, qe, dtype=jnp.int32)
        idx_dev, d2_dev = knn_one_qblock(Q, q_ids)
        idx_blk, d2_blk = jax.device_get((idx_dev, d2_dev))
        out_idx[qs:qe] = idx_blk
        out_d2[qs:qe] = d2_blk

    if not self_case:
        indptr = np.arange(0, n_q * k + 1, k, dtype=np.int64)
        return csr_matrix(
            (out_d2.reshape(-1), out_idx.reshape(-1), indptr),
            shape=(n_q, n_r),
            dtype=np.float32,
        )

    return _symmetrize_union_min_d2(n_q, out_idx, out_d2)
