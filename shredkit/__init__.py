"""shredkit: parse, verify, recover and deshred Solana Merkle shreds.

Copyright (c) 2026 Norbert Jarzmik. MIT License (see LICENSE).
"""

from .entries import Entry, EntryBatch, batch_ranges, deshred, parse_entry_batch, serialize_entries
from .errors import DecodeError, RecoveryError, ShredError, ShredkitError
from .fec import RecoveredSet, group_by_set, recover_set
from .layout import Shred, Variant, parse_shred
from .merkle import merkle_root, verify_leader_signature
from .tx import Transaction, parse_transaction

__version__ = "0.1.0"

__all__ = [
    "DecodeError", "Entry", "EntryBatch", "RecoveredSet", "RecoveryError", "Shred", "ShredError", "ShredkitError",
    "Transaction", "Variant", "batch_ranges", "deshred", "group_by_set", "merkle_root", "parse_entry_batch",
    "parse_shred", "parse_transaction", "recover_set", "serialize_entries", "verify_leader_signature",
]
