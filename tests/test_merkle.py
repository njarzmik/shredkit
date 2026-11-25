import random

import pytest
from conftest import patch

from shredkit.errors import ShredError
from shredkit.layout import parse_shred
from shredkit.merkle import hash_leaf, make_proof, merkle_root, root_from_proof, tree_levels, verify_leader_signature
from shredkit.synth import make_key
from shredkit.tx import public_key_bytes


def _rejected(raw: bytes, pub: bytes) -> bool:
    """A shred is rejected if it no longer parses, its proof no longer fits, or the signature fails."""
    try:
        return not verify_leader_signature(parse_shred(raw), pub)
    except ShredError:
        return True


def test_each_set_proves_one_root_and_sets_are_chained(batch):
    first, second = batch.shreds[:64], batch.shreds[64:]
    root1 = {merkle_root(s) for s in first}
    root2 = {merkle_root(s) for s in second}
    assert len(root1) == len(root2) == 1 and root1 != root2
    assert root2 == {batch.built.merkle_root}
    assert {s.chained_root() for s in second} == root1            # set 2 carries set 1's root
    assert {s.chained_root() for s in first} == {bytes(32)}


def test_leader_signature_verifies_on_every_shred(batch, resigned_batch):
    for b in (batch, resigned_batch):
        assert all(verify_leader_signature(s, b.leader_pub) for s in b.shreds)


def test_wrong_leader_key_fails_on_every_shred(batch):
    other = public_key_bytes(make_key(random.Random(99)))
    assert not any(verify_leader_signature(s, other) for s in batch.shreds)


@pytest.mark.parametrize("which", [5, 40])           # one data shred, one code shred
def test_every_signed_byte_is_covered(batch, which):
    """Flip one bit in each byte of [0, proof end): every single change must be rejected."""
    raw = batch.built.shreds[which]
    end = parse_shred(raw).variant.retransmitter_offset
    assert end == len(raw)                            # not resigned: everything is signed
    for pos in range(end):
        buf = bytearray(raw)
        buf[pos] ^= 1 << (pos % 8)
        assert _rejected(bytes(buf), batch.leader_pub), pos


def test_retransmitter_signature_is_outside_the_signed_bytes(resigned_batch):
    """The control in the other direction: bytes the leader does not sign must not affect the check."""
    shred = resigned_batch.shreds[0]
    start = shred.variant.retransmitter_offset
    assert len(shred.payload) - start == 64
    buf = bytearray(shred.payload)
    buf[start:] = bytes(64)
    assert verify_leader_signature(parse_shred(bytes(buf)), resigned_batch.leader_pub)


def test_shard_index_outside_the_tree_is_rejected(batch):
    """Index - fec_set_index = 64 does not fit a 6-level proof; it must not verify by accident."""
    raw = patch(batch.built.shreds[0], 73, "<I", 64)
    with pytest.raises(ShredError, match="does not fit"):
        merkle_root(parse_shred(raw))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 64])
def test_proofs_lead_to_the_root_for_any_leaf_count(n):
    leaves = [hash_leaf(bytes([i])) for i in range(n)]
    levels = tree_levels(leaves)
    for i in range(n):
        assert root_from_proof(leaves[i], i, make_proof(levels, i)) == levels[-1][0]
