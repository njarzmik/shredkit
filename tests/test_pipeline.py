"""End to end: a signed synthetic slot -> verify -> drop half of every set -> recover -> deshred -> parse."""

import random

import pytest

from shredkit.demo import run_demo
from shredkit.entries import batch_ranges, deshred
from shredkit.fec import group_by_set, recover_set
from shredkit.layout import parse_shred
from shredkit.merkle import verify_leader_signature
from shredkit.synth import make_key, synthetic_slot
from shredkit.tx import public_key_bytes


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_signed_slot_survives_losing_half_of_every_set(seed):
    rng = random.Random(seed)
    leader = make_key(rng)
    pub = public_key_bytes(leader)
    syn = synthetic_slot(rng, leader, 77_000, n_batches=3, zero_padding={0: 1_000})
    shreds = [parse_shred(p) for p in syn.payloads]
    assert all(verify_leader_signature(s, pub) for s in shreds)

    data = {}
    for members in group_by_set(shreds).values():
        recovered = recover_set(rng.sample(members, 32))
        assert all(verify_leader_signature(s, pub) for s in recovered.data_shreds)
        data.update((s.index, s) for s in recovered.data_shreds)

    complete, incomplete = batch_ranges(data, [i for i, s in data.items() if s.data_complete])
    assert incomplete == 0 and len(complete) == 3
    batches = [deshred([data[i] for i in span]) for span in complete]
    assert [b.status for b in batches] == ["zero_padded", "exact", "exact"]
    txs = [t for b in batches for t in b.transactions]
    assert [t.raw for t in txs] == syn.transactions
    assert {t.version for t in txs} == {"legacy", "v0", "v1"}
    assert all(t.verify_signatures() for t in txs)


def test_demo_reports_success():
    lines: list[str] = []
    assert run_demo(seed=4, out=lines.append)
    assert any("zero_padded" in line for line in lines)
