import random

import numpy as np
import pytest

from shredkit import rs
from shredkit.errors import RecoveryError


def test_known_products_in_the_field():
    """Hand-checked products in GF(2^8) mod x^8 + x^4 + x^3 + x^2 + 1 (0x11D). 2 * 128 = 0x100 reduces to 29,
    which pins the polynomial; the others exercise multi-term reductions."""
    assert rs.gf_mul(2, 128) == 29
    assert rs.gf_mul(3, 4) == 12
    assert rs.gf_mul(7, 7) == 21
    assert rs.gf_mul(23, 45) == 41
    assert all(rs.gf_mul(a, rs.gf_inv(a)) == 1 for a in range(1, 256))
    assert rs.gf_pow(0, 0) == 1 and rs.gf_pow(0, 3) == 0 and rs.gf_pow(2, 8) == 29


def test_two_data_one_parity_by_hand():
    """k = 2, m = 1: V = [[1, 0], [1, 1], [1, 2]], inverse(V[:2]) = V[:2] (over GF(2) 1 + 1 = 0), so the parity
    row is [1, 2] * [[1, 0], [1, 1]] = [1 ^ 2, 2] = [3, 2]: parity = 3 * d0 + 2 * d1."""
    d0, d1 = bytes([1, 2, 200, 0]), bytes([9, 0, 17, 255])
    expected = bytes(rs.gf_mul(3, a) ^ rs.gf_mul(2, b) for a, b in zip(d0, d1))
    assert rs.encode([d0, d1], 1) == [expected]


def test_coding_matrix_is_systematic():
    m = rs.coding_matrix(32, 32)
    assert m.shape == (64, 32)
    assert (m[:32] == np.eye(32, dtype=np.uint8)).all()


def _data(rng: random.Random, k: int = 32, length: int = 987) -> list[bytes]:
    return [rng.randbytes(length) for _ in range(k)]


@pytest.mark.parametrize("pattern", ["parity_only", "random", "alternate", "one_data"])
def test_any_32_of_64_shards_recover_the_data(pattern):
    rng = random.Random(pattern)
    data = _data(rng)
    shards = dict(enumerate(data + rs.encode(data, 32)))
    keep = {"parity_only": range(32, 64), "alternate": range(0, 64, 2), "one_data": [5, *range(33, 64)],
            "random": rng.sample(range(64), 32)}[pattern]
    assert rs.recover_data({i: shards[i] for i in keep}, 32, 32) == data


def test_31_shards_are_not_enough():
    rng = random.Random(1)
    data = _data(rng)
    shards = dict(enumerate(data + rs.encode(data, 32)))
    with pytest.raises(RecoveryError, match="need 32"):
        rs.recover_data({i: shards[i] for i in range(1, 32)}, 32, 32)


def test_bad_shard_index_and_length_are_rejected():
    rng = random.Random(2)
    data = _data(rng, k=4, length=8)
    with pytest.raises(RecoveryError, match="out of range"):
        rs.recover_data({0: data[0], 9: data[1]}, 4, 4)
    with pytest.raises(RecoveryError, match="length"):
        rs.encode([data[0], data[1][:7], data[2], data[3]], 4)
