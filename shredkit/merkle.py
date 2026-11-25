"""Merkle tree over an erasure set, and the leader signature on its root.

Agave v4.4.0 ledger/src/shred/merkle_tree.rs: domain prefixes (:17-18), parent = hash of the two children's
first 20 bytes (:111-115), tree construction (:53-74), proof (:82-107), root from a proof (:120-137).
ledger/src/shred/merkle.rs: the leaf covers b[64:proof_offset) (:110, :174, :583-587).

Every shred of a set (32 data + 32 code) is one leaf. The leader signs only the 32-byte root, once per set, and
every shred carries that signature plus its own proof. So one shred on its own is enough to check that the
leader produced it: hash the leaf, walk the proof up to the root, verify the signature over the root.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .errors import ShredError
from .layout import PROOF_ENTRY_SIZE, Shred

LEAF_PREFIX = b"\x00SOLANA_MERKLE_SHREDS_LEAF"
NODE_PREFIX = b"\x01SOLANA_MERKLE_SHREDS_NODE"


def hash_leaf(leaf_bytes: bytes) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + leaf_bytes).digest()


def join_nodes(left: bytes, right: bytes) -> bytes:
    """Only 20 bytes of each child enter the parent hash, which is why proof entries on the wire are 20 bytes.
    The distinct leaf and node prefixes stop a leaf from being passed off as an inner node."""
    return hashlib.sha256(NODE_PREFIX + left[:PROOF_ENTRY_SIZE] + right[:PROOF_ENTRY_SIZE]).digest()


def tree_levels(leaves: list[bytes]) -> list[list[bytes]]:
    """All levels, leaves first, root last. An odd node at the end of a level is paired with itself."""
    if not leaves:
        raise ValueError("a Merkle tree needs at least one leaf")
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([join_nodes(cur[j], cur[min(j + 1, len(cur) - 1)]) for j in range(0, len(cur), 2)])
    return levels


def make_proof(levels: list[list[bytes]], index: int) -> list[bytes]:
    """The 20-byte siblings from leaf `index` up to (not including) the root."""
    proof = []
    for level in levels[:-1]:
        proof.append(level[min(index ^ 1, len(level) - 1)][:PROOF_ENTRY_SIZE])
        index >>= 1
    return proof


def root_from_proof(leaf: bytes, index: int, proof: list[bytes]) -> bytes:
    """Walk up: an even index means our node is the left child. An index that does not fit in the proof's
    depth is rejected, otherwise a shred could claim a position outside the tree."""
    node = leaf
    for sibling in proof:
        node = join_nodes(node, sibling) if index % 2 == 0 else join_nodes(sibling, node)
        index >>= 1
    if index != 0:
        raise ShredError("shard index does not fit the Merkle proof")
    return node


def merkle_root(shred: Shred) -> bytes:
    """The root this shred proves, from its own bytes only."""
    if shred.shard_index < 0:
        raise ShredError("negative shard index")
    return root_from_proof(hash_leaf(shred.merkle_leaf_bytes()), shred.shard_index, shred.proof())


@lru_cache(maxsize=1024)
def _public_key(raw: bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(raw)


def verify_leader_signature(shred: Shred, leader_pubkey: bytes) -> bool:
    """True if the shred's signature is the leader's Ed25519 signature over the root its proof leads to.

    Returns False for a wrong signature, wrong key or any altered byte in [0, proof end). Raises ShredError if
    the proof cannot be walked at all, so malformed input is counted apart from forged input. The retransmitter
    signature of resigned shreds is outside the signed bytes and is not checked.
    """
    root = merkle_root(shred)
    try:
        _public_key(bytes(leader_pubkey)).verify(shred.signature, root)
    except InvalidSignature:
        return False
    return True
