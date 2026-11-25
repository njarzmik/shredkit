import random

import pytest

from shredkit.layout import parse_shred
from shredkit.rawrec import RunStats, deshred_run, write_run
from shredkit.synth import make_key, synthetic_slot


@pytest.fixture(scope="module")
def slots():
    rng = random.Random(21)
    leader = make_key(rng)
    return [synthetic_slot(rng, leader, 3_000 + k, n_batches=3, zero_padding={2: 300}) for k in range(3)]


def _run(tmp_path, payloads, **kw):
    write_run(tmp_path, payloads)
    stats = RunStats()
    return list(deshred_run(tmp_path, stats=stats, **kw)), stats


def _txs(results):
    return [t.raw for r in results for b in r.batches for t in b.batch.transactions]


def test_a_clean_run_deshreds_completely(tmp_path, slots):
    results, stats = _run(tmp_path, [p for s in slots for p in s.payloads])
    assert [r.slot for r in results] == [3_000, 3_001, 3_002]
    assert all(r.incomplete_batches == 0 and len(r.batches) == 3 for r in results)
    assert [b.batch.status for b in results[0].batches] == ["exact", "exact", "zero_padded"]
    assert _txs(results) == [t for s in slots for t in s.transactions]
    assert stats.payload_records == sum(1 for s in slots for p in s.payloads if parse_shred(p).is_data)


def test_arrival_order_does_not_matter(tmp_path, slots):
    payloads = list(slots[0].payloads)
    random.Random(5).shuffle(payloads)
    results, _ = _run(tmp_path, payloads)
    assert _txs(results) == slots[0].transactions


def test_a_lost_shred_leaves_only_its_batch_incomplete(tmp_path, slots):
    payloads = list(slots[0].payloads)
    lost = next(p for p in payloads if parse_shred(p).is_data and parse_shred(p).index == 70)
    payloads.remove(lost)
    (result,), _ = _run(tmp_path, payloads)
    assert result.incomplete_batches == 1
    assert [(b.first_index, b.last_index) for b in result.batches] == [(0, 63), (96, 127)]


def test_duplicates_late_payloads_and_a_torn_tail(tmp_path, slots):
    first, rest = slots[0].payloads, slots[1].payloads + slots[2].payloads
    last_data = [p for p in first if parse_shred(p).is_data][-1]
    payloads = [p for p in first if p is not last_data] + [first[0]] + rest + [last_data]
    write_run(tmp_path, payloads)
    pay = next(tmp_path.glob("pay_*.bin"))
    pay.write_bytes(pay.read_bytes()[:-10])            # cut the last record: a crash mid-write
    stats = RunStats()
    results = list(deshred_run(tmp_path, window=1, stats=stats))
    assert stats.duplicate_payloads == 1 and stats.conflicting_payloads == 0
    assert stats.torn_bytes > 0
    assert stats.late_payloads == 0                     # the torn record was the late one; it never arrived
    assert results[0].slot == 3_000 and results[0].incomplete_batches == 1


def test_a_late_payload_is_counted_not_merged(tmp_path, slots):
    first, rest = slots[0].payloads, slots[1].payloads + slots[2].payloads
    last_data = [p for p in first if parse_shred(p).is_data][-1]
    payloads = [p for p in first if p is not last_data] + rest + [last_data]
    results, stats = _run(tmp_path, payloads, window=1)
    assert stats.late_payloads == 1
    assert results[0].incomplete_batches == 1


def test_missing_record_files_are_an_error(tmp_path, slots):
    write_run(tmp_path, slots[0].payloads)
    next(tmp_path.glob("rec_*.bin")).unlink()
    with pytest.raises(FileNotFoundError, match="rec_"):
        list(deshred_run(tmp_path))


def test_not_a_rawrec_file_is_an_error(tmp_path, slots):
    write_run(tmp_path, slots[0].payloads)
    pay = next(tmp_path.glob("pay_*.bin"))
    pay.write_bytes(b"NOTRAWRC" + pay.read_bytes()[8:])
    with pytest.raises(ValueError, match="not a rawrec file"):
        list(deshred_run(tmp_path))
