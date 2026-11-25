"""The whole pipeline on one synthetic slot: build, verify, drop, recover, deshred, parse, compare.

Every step compares against what was built, so the demo doubles as an end-to-end self-check: it returns False if
any count is off.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable

from .entries import batch_ranges, deshred
from .errors import ShredError
from .fec import group_by_set, recover_set
from .layout import Shred, parse_shred
from .merkle import verify_leader_signature
from .synth import make_key, synthetic_slot
from .tx import public_key_bytes

SLOT = 1_000


def run_demo(seed: int = 1, drop: int = 32, out: Callable[[str], None] = print) -> bool:
    if not 0 <= drop <= 32:
        raise ValueError("drop must be 0..32: an erasure set of 32 + 32 survives the loss of any 32")
    rng = random.Random(seed)
    leader = make_key(rng)
    syn = synthetic_slot(rng, leader, SLOT, n_batches=3, zero_padding={1: 700})
    shreds = [parse_shred(p) for p in syn.payloads]
    sets = group_by_set(shreds)
    n_data = sum(s.is_data for s in shreds)
    out(f"built slot {SLOT}: {len(syn.batch_sizes)} batches, {len(sets)} erasure sets, {len(shreds)} shreds "
        f"({n_data} data, {len(shreds) - n_data} code)")
    ok = _check_signatures(shreds, public_key_bytes(leader), public_key_bytes(make_key(rng)), rng, out)
    data, rebuilt = _drop_and_recover(sets, drop, rng)
    originals = {s.index: s for s in shreds if s.is_data}
    same = all(_signed_bytes(data[i]) == _signed_bytes(originals[i]) for i in originals)
    out(f"dropped {drop} of 64 shreds in each set, rebuilt {rebuilt} data shreds; "
        f"identical to the originals: {same}")
    parsed = _deshred_all(data, out)
    same_txs = [t.raw for t in parsed] == syn.transactions
    sigs = all(t.verify_signatures() for t in parsed)
    out(f"transactions: {len(parsed)} parsed, identical to the ones built: {same_txs}, signatures verify: {sigs}")
    return ok and same and same_txs and sigs


def _check_signatures(shreds: list[Shred], leader: bytes, other: bytes, rng: random.Random,
                      out: Callable[[str], None]) -> bool:
    good = sum(verify_leader_signature(s, leader) for s in shreds)
    flipped = sum(_verifies_after_flip(s, leader, rng) for s in shreds)
    wrong = sum(verify_leader_signature(s, other) for s in shreds)
    out(f"leader signature: {good} / {len(shreds)} verify; after one random bit flip each: {flipped}; "
        f"with another key: {wrong}")
    return good == len(shreds) and flipped == 0 and wrong == 0


def _verifies_after_flip(shred: Shred, leader: bytes, rng: random.Random) -> bool:
    """Flip one bit somewhere in the signed bytes. Any outcome other than True (bad signature, unparseable
    header, proof that no longer fits) is a rejection."""
    buf = bytearray(shred.payload)
    buf[rng.randrange(shred.variant.retransmitter_offset)] ^= 1 << rng.randrange(8)
    try:
        return verify_leader_signature(parse_shred(bytes(buf)), leader)
    except ShredError:
        return False


def _drop_and_recover(sets: dict[tuple[int, int], list[Shred]], drop: int,
                      rng: random.Random) -> tuple[dict[int, Shred], int]:
    data: dict[int, Shred] = {}
    rebuilt = 0
    for _, members in sorted(sets.items()):
        recovered = recover_set(rng.sample(members, len(members) - drop))
        rebuilt += len(recovered.recovered_indices)
        data.update((s.index, s) for s in recovered.data_shreds)
    return data, rebuilt


def _signed_bytes(shred: Shred) -> bytes:
    """Everything except a resigned shred's retransmitter signature, which recovery cannot rebuild."""
    return shred.payload[:shred.variant.retransmitter_offset]


def _deshred_all(data: dict[int, Shred], out: Callable[[str], None]) -> list:
    ends = [i for i, s in data.items() if s.data_complete]
    complete, _ = batch_ranges(data, ends)
    parsed = []
    for span in complete:
        batch = deshred([data[i] for i in span])
        versions = Counter(t.version for t in batch.transactions)
        tail = f", {batch.tail_length} zero bytes after the entries" if batch.status == "zero_padded" else ""
        out(f"  batch shreds {span.start}-{span.stop - 1}: {len(batch.entries)} entries, "
            f"{len(batch.transactions)} txs {dict(sorted(versions.items()))}, {batch.status}{tail}")
        parsed += batch.transactions
    return parsed
