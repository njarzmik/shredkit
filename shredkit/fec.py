"""Erasure-set recovery: rebuild missing data shreds and prove the rebuilt bytes are the leader's.

Reed-Solomon alone would turn any 32 shards into 32 data shards, including wrong ones if a shard was corrupted
or belonged to another set. So recovery here is checked the same way the validator checks it: after decoding,
re-encode the parity, rebuild every shred of the set byte for byte (headers, chained root, proofs), recompute
the Merkle root over all of them, and require it to equal the root the received shreds prove. That root is what
the leader signed, so recovered shreds are exactly as trustworthy as received ones and pass the same
verify_leader_signature check.

The one thing that cannot be rebuilt is the retransmitter signature of resigned sets: it is outside both the
erasure shard and the Merkle leaf. Recovered shreds carry zeros there.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from . import rs
from .errors import RecoveryError
from .layout import SHREDS_PER_SET, Shred, assemble_payload, code_header, parse_shred
from .merkle import hash_leaf, make_proof, merkle_root, tree_levels


@dataclass(frozen=True)
class RecoveredSet:
    data_shreds: tuple[Shred, ...]       # every data shred of the set, in index order
    recovered_indices: tuple[int, ...]   # shred indexes that were rebuilt rather than received
    merkle_root: bytes


def group_by_set(shreds: Iterable[Shred]) -> dict[tuple[int, int], list[Shred]]:
    """{(slot, fec_set_index): shreds}."""
    sets: dict[tuple[int, int], list[Shred]] = defaultdict(list)
    for s in shreds:
        sets[(s.slot, s.fec_set_index)].append(s)
    return dict(sets)


def recover_set(shreds: Sequence[Shred]) -> RecoveredSet:
    """All data shreds of one erasure set from any `num_data` of its shreds. Raises RecoveryError if the shreds
    are inconsistent, too few, or if the rebuilt set does not reproduce the signed root."""
    shreds = _dedupe(shreds)
    root = _common_root(shreds)
    num_data, num_coding, code_base = _set_shape(shreds)
    data = {s.shard_index: s for s in shreds if s.is_data}
    if any(i >= num_data for i in data):
        raise RecoveryError("a data shred's position is outside the set")
    if len(data) == num_data:
        return RecoveredSet(tuple(data[i] for i in range(num_data)), (), root)
    if code_base is None or len(shreds) < num_data:
        raise RecoveryError(f"need {num_data} shreds of the set, have {len(shreds)}")
    shards = {s.shard_index: s.erasure_shard() for s in shreds}
    data_shards = rs.recover_data(shards, num_data, num_coding)
    rebuilt = _rebuild(shreds[0], data_shards, num_coding, code_base, root)
    missing = [i for i in range(num_data) if i not in data]
    out = tuple(data[i] if i in data else rebuilt[i] for i in range(num_data))
    return RecoveredSet(out, tuple(out[i].index for i in missing), root)


def _dedupe(shreds: Sequence[Shred]) -> list[Shred]:
    if not shreds:
        raise RecoveryError("no shreds")
    seen: dict[int, Shred] = {}
    for s in shreds:
        prev = seen.setdefault(s.shard_index, s)
        if prev.payload != s.payload:
            raise RecoveryError(f"two different shreds at shard index {s.shard_index}")
    return list(seen.values())


def _common_root(shreds: list[Shred]) -> bytes:
    """One set = one slot, one fec_set_index, one variant family, one proven root, one leader signature."""
    keys = {(s.slot, s.fec_set_index, s.version, s.variant.resigned, s.variant.proof_size) for s in shreds}
    if len(keys) != 1:
        raise RecoveryError("shreds from different sets or variant families")
    roots = {merkle_root(s) for s in shreds}
    if len(roots) != 1:
        raise RecoveryError(f"shreds prove {len(roots)} different Merkle roots")
    if len({s.signature for s in shreds}) != 1:
        raise RecoveryError("shreds carry different signatures")
    if len({s.chained_root() for s in shreds}) != 1:
        raise RecoveryError("shreds carry different chained roots")
    return roots.pop()


def _set_shape(shreds: list[Shred]) -> tuple[int, int, int | None]:
    """(num_data, num_coding, index of the first code shred). Only code headers carry the counts; without any code
    shred the set is assumed to be the standard 32:32, which matters only when nothing needs recovering."""
    codes = [s for s in shreds if not s.is_data]
    if not codes:
        return SHREDS_PER_SET, SHREDS_PER_SET, None
    shapes = {(c.num_data, c.num_coding, c.index - c.position) for c in codes}
    if len(shapes) != 1:
        raise RecoveryError("code shreds disagree on the set's shape")
    return shapes.pop()


def _rebuild(template: Shred, data_shards: list[bytes], num_coding: int, code_base: int,
             root: bytes) -> list[Shred]:
    """Every shred of the set from the decoded data shards; checks the rebuilt root; returns the data shreds."""
    num_data = len(data_shards)
    code_variant = template.variant.sibling(False)
    data_variant = template.variant.sibling(True)
    chained = template.chained_root()
    parity = rs.encode(data_shards, num_coding)
    code_bodies = [code_header(code_variant, template.slot, code_base + j, template.version,
                               template.fec_set_index, num_data, num_coding, j) + parity[j]
                   for j in range(num_coding)]
    bodies = list(data_shards) + code_bodies
    levels = tree_levels([hash_leaf(b + chained) for b in bodies])
    if levels[-1][0] != root:
        raise RecoveryError("rebuilt shreds do not reproduce the signed Merkle root")
    out = []
    for i, body in enumerate(data_shards):
        payload = assemble_payload(data_variant, template.signature, body, chained, make_proof(levels, i))
        out.append(parse_shred(payload))
    return out
