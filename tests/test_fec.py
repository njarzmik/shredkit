import random

import pytest
from conftest import make_signed_batch

from shredkit import fec
from shredkit.errors import RecoveryError
from shredkit.fec import group_by_set, recover_set
from shredkit.merkle import verify_leader_signature


def _first_set(b):
    return group_by_set(b.shreds)[(5_000, 0)]


@pytest.mark.parametrize("pattern", ["all_data_lost", "random_half", "alternate"])
def test_recovery_after_losing_half_is_byte_exact_and_verifies(batch, pattern):
    members = _first_set(batch)
    rng = random.Random(pattern)
    keep = {"all_data_lost": members[32:], "random_half": rng.sample(members, 32),
            "alternate": members[::2]}[pattern]
    out = recover_set(keep)
    assert [s.payload for s in out.data_shreds] == [s.payload for s in members[:32]]
    assert len(out.recovered_indices) == 32 - sum(s.is_data for s in keep)
    assert all(verify_leader_signature(s, batch.leader_pub) for s in out.data_shreds)


def test_resigned_set_recovers_all_but_the_retransmitter_signature(resigned_batch):
    members = group_by_set(resigned_batch.shreds)[(5_000, 0)]
    out = recover_set(members[32:])
    for got, want in zip(out.data_shreds, members[:32]):
        cut = want.variant.retransmitter_offset
        assert got.payload[:cut] == want.payload[:cut]
        assert got.payload[cut:] == bytes(64) != want.payload[cut:]
        assert verify_leader_signature(got, resigned_batch.leader_pub)


def test_nothing_to_recover_returns_the_received_shreds(batch):
    out = recover_set(_first_set(batch)[:32])
    assert out.recovered_indices == ()


def test_33_lost_shreds_cannot_be_recovered(batch):
    with pytest.raises(RecoveryError, match="need 32"):
        recover_set(_first_set(batch)[33:])


def test_shreds_of_another_set_or_leader_are_rejected(batch):
    other_leader = make_signed_batch(7)             # same slot and indexes, different leader and data
    mixed = _first_set(batch)[:20] + group_by_set(other_leader.shreds)[(5_000, 0)][40:]
    with pytest.raises(RecoveryError, match="different sets|different Merkle roots"):
        recover_set(mixed)
    with pytest.raises(RecoveryError, match="different sets"):
        recover_set(_first_set(batch)[:20] + batch.shreds[64 + 40:])


def test_two_different_shreds_at_one_position_are_rejected(batch):
    members = _first_set(batch)
    imposter = make_signed_batch(8).shreds[33]       # a code shred at the same shard index as members[33]
    with pytest.raises(RecoveryError, match="two different shreds"):
        recover_set(members[33:] + [imposter])


def test_a_wrong_decode_is_caught_by_the_root_check(batch, monkeypatch):
    """Planted error: corrupt one byte of what Reed-Solomon returns. Recovery must refuse, because the rebuilt
    shreds no longer hash to the root the leader signed. Without this check the bad bytes would pass through."""
    real = fec.rs.recover_data

    def corrupted(*args):
        shards = real(*args)
        shards[3] = shards[3][:100] + bytes([shards[3][100] ^ 0x01]) + shards[3][101:]
        return shards

    monkeypatch.setattr(fec.rs, "recover_data", corrupted)
    with pytest.raises(RecoveryError, match="do not reproduce the signed Merkle root"):
        recover_set(_first_set(batch)[32:])
