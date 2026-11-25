"""Reed-Solomon erasure coding over GF(2^8), built to match the reed-solomon-erasure crate 6.0.0 that Agave uses
(agave v4.4.0 Cargo.toml:311 pins "6.0.0"; shredder.rs:6 imports its galois_8 field).

The parts that must match the crate exactly, or the parity bytes differ:

  field     GF(2^8) with generating polynomial 29, i.e. x^8 + x^4 + x^3 + x^2 + 1 (build.rs:11-25); field element n
            is the byte n (galois_8.rs:37-39)
  matrix    vandermonde[r][c] = r^c for r in 0..k+m (matrix.rs:263-276); coding matrix = V * inverse(V[:k])
            (core.rs:430-436), so its top k rows are the identity: data shards pass through unchanged
  encode    parity row i = sum over j of matrix[k + i][j] * data[j], byte by byte
  recover   take any k present rows of the coding matrix, invert that k x k matrix, multiply by the k present
            shards; any k rows of a Vandermonde-derived matrix are invertible, so any k of k + m shards suffice

Byte-wise arithmetic is vectorised with numpy: one gather from a 256 x 256 product table, then an XOR reduce.
The small k x k inversions stay in plain Python.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .errors import RecoveryError

POLYNOMIAL = 29


def _build_tables() -> tuple[list[int], list[int], np.ndarray]:
    exp = [0] * 510
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = exp[i + 255] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x = (x & 0xFF) ^ POLYNOMIAL
    log_arr = np.array(log, dtype=np.int32)
    mul = np.zeros((256, 256), dtype=np.uint8)
    mul[1:, 1:] = np.array(exp, dtype=np.uint8)[log_arr[1:, None] + log_arr[None, 1:]]
    return exp, log, mul


GF_EXP, GF_LOG, GF_MUL = _build_tables()


def gf_mul(a: int, b: int) -> int:
    return int(GF_MUL[a, b])


def gf_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in GF(2^8)")
    return GF_EXP[255 - GF_LOG[a]]


def gf_pow(a: int, n: int) -> int:
    """a^n with 0^0 = 1, as the crate's galois_8::exp (the Vandermonde matrix needs row 0 = [1, 0, 0, ...])."""
    if n == 0:
        return 1
    if a == 0:
        return 0
    return GF_EXP[(GF_LOG[a] * n) % 255]


def gf_invert_matrix(m: list[list[int]]) -> list[list[int]]:
    """Gauss-Jordan elimination over GF(2^8). Raises RecoveryError on a singular matrix."""
    n = len(m)
    a = [list(row) + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        pivot = next((r for r in range(c, n) if a[r][c]), None)
        if pivot is None:
            raise RecoveryError("singular matrix")
        a[c], a[pivot] = a[pivot], a[c]
        inv = gf_inv(a[c][c])
        a[c] = [gf_mul(v, inv) for v in a[c]]
        for r in range(n):
            if r != c and a[r][c]:
                f = a[r][c]
                a[r] = [v ^ gf_mul(f, w) for v, w in zip(a[r], a[c])]
    return [row[n:] for row in a]


def gf_matmul(x: list[list[int]], y: list[list[int]]) -> list[list[int]]:
    out = []
    for row in x:
        new = []
        for c in range(len(y[0])):
            acc = 0
            for k, v in enumerate(row):
                acc ^= gf_mul(v, y[k][c])
            new.append(acc)
        out.append(new)
    return out


@lru_cache(maxsize=16)
def coding_matrix(num_data: int, num_parity: int) -> np.ndarray:
    """(k + m) x k matrix V * inverse(V[:k]); rows k.. produce parity. Cached: Solana sets are almost all 32:32."""
    if num_data < 1 or num_parity < 0 or num_data + num_parity > 256:
        raise RecoveryError(f"no GF(2^8) code for {num_data} data + {num_parity} parity shards")
    v = [[gf_pow(r, c) for c in range(num_data)] for r in range(num_data + num_parity)]
    m = np.array(gf_matmul(v, gf_invert_matrix(v[:num_data])), dtype=np.uint8)
    m.flags.writeable = False
    return m


def _apply(coefficients: np.ndarray, shards: np.ndarray) -> np.ndarray:
    """(r x k) coefficients times (k x L) shards over GF(2^8) -> (r x L)."""
    products = GF_MUL[coefficients[:, :, None], shards[None, :, :]]
    return np.bitwise_xor.reduce(products, axis=1)


def _stack(shards: list[bytes]) -> np.ndarray:
    if len({len(s) for s in shards}) != 1:
        raise RecoveryError("shards differ in length")
    return np.stack([np.frombuffer(s, dtype=np.uint8) for s in shards])


def encode(data_shards: list[bytes], num_parity: int) -> list[bytes]:
    """Parity shards for `data_shards`, as ReedSolomon::encode."""
    k = len(data_shards)
    parity = _apply(coding_matrix(k, num_parity)[k:], _stack(data_shards))
    return [row.tobytes() for row in parity]


def recover_data(shards: dict[int, bytes], num_data: int, num_parity: int) -> list[bytes]:
    """All `num_data` data shards from any `num_data` of the shards {shard index: bytes} (data 0..k-1, parity
    k..k+m-1). Data shards are preferred when choosing rows, so the fewest parity rows enter the inversion."""
    bad = [i for i in shards if not 0 <= i < num_data + num_parity]
    if bad:
        raise RecoveryError(f"shard index out of range: {bad[:3]}")
    if all(i in shards for i in range(num_data)):
        return [shards[i] for i in range(num_data)]
    if len(shards) < num_data:
        raise RecoveryError(f"need {num_data} shards, have {len(shards)}")
    rows = sorted(shards)[:num_data]
    sub = coding_matrix(num_data, num_parity)[rows]
    inverse = np.array(gf_invert_matrix(sub.tolist()), dtype=np.uint8)
    data = _apply(inverse, _stack([shards[i] for i in rows]))
    return [row.tobytes() for row in data]
