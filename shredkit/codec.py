"""Bounds-checked byte reading, Solana's compact-u16, and base58 for display.

Every decoder in shredkit reads through `ByteReader`, so a truncated or lying length field raises `DecodeError`
at the read that runs past the end, instead of producing a short slice that Python would silently accept.
"""

from __future__ import annotations

import struct

from .errors import DecodeError

B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


class ByteReader:
    """A cursor over a bytes object. All reads are little-endian, as in bincode and the shred headers."""

    def __init__(self, buf: bytes, pos: int = 0) -> None:
        self.buf = buf
        self.pos = pos

    def remaining(self) -> int:
        return len(self.buf) - self.pos

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if n < 0 or end > len(self.buf):
            raise DecodeError(f"truncated: need {n} bytes at offset {self.pos}, have {self.remaining()}")
        out = bytes(self.buf[self.pos:end])
        self.pos = end
        return out

    def peek_u8(self) -> int:
        if self.pos >= len(self.buf):
            raise DecodeError(f"truncated: need 1 byte at offset {self.pos}")
        return self.buf[self.pos]

    def u8(self) -> int:
        value = self.peek_u8()
        self.pos += 1
        return value

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def compact_u16(self) -> int:
        """Solana's short_vec length: 7 bits per byte, high bit = more, at most 3 bytes.

        Non-canonical encodings (a trailing zero byte) and values over 0xFFFF are rejected, as the validator's
        decoder does, so one transaction cannot have two valid byte forms.
        """
        value = 0
        for i in range(3):
            byte = self.u8()
            value |= (byte & 0x7F) << (7 * i)
            if not byte & 0x80:
                if byte == 0 and i > 0:
                    raise DecodeError("non-canonical compact-u16")
                if value > 0xFFFF:
                    raise DecodeError("compact-u16 overflow")
                return value
        raise DecodeError("compact-u16 longer than 3 bytes")


def encode_compact_u16(n: int) -> bytes:
    if not 0 <= n <= 0xFFFF:
        raise ValueError(f"compact-u16 out of range: {n}")
    out = bytearray()
    while True:
        low = n & 0x7F
        n >>= 7
        if n:
            out.append(low | 0x80)
        else:
            out.append(low)
            return bytes(out)


def b58encode(raw: bytes) -> str:
    """Base58 (Bitcoin alphabet), used only to print signatures and keys the way explorers show them."""
    n = int.from_bytes(raw, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = B58_ALPHABET[r] + out
    leading_zeros = len(raw) - len(raw.lstrip(b"\x00"))
    return "1" * leading_zeros + out
