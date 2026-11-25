"""Exception types.

Each stage has its own error so a caller can count failures by cause: a shred that does not parse, a set that
cannot be recovered and a batch whose bytes do not decode are different problems with different fixes. A bad
signature is not an exception at all: `verify_leader_signature` returns False for it.
"""


class ShredkitError(ValueError):
    """Base class for every error raised by shredkit."""


class ShredError(ShredkitError):
    """A shred is truncated, has an unsupported variant, a header field out of range, or a proof that does not fit."""


class RecoveryError(ShredkitError):
    """An erasure set cannot be recovered: too few shards, inconsistent shreds, or a rebuilt root that does not match."""


class DecodeError(ShredkitError):
    """An entry batch or a transaction does not decode."""
