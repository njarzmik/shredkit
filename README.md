# shredkit

Solana Merkle shreds in Python: parse them, verify the leader's signature, recover lost ones with Reed-Solomon, and deshred them back into entries and transactions.

A Solana leader does not send blocks. It sends shreds: UDP packets of about 1.2 KB, grouped into erasure sets of 32 data and 32 coding shreds, each set signed once through a Merkle root. shredkit implements that format from the validator's source code. Given shred bytes (from a packet capture, or from my recorder [rawrec](https://github.com/njarzmik/rawrec)) it checks that the leader signed them, rebuilds missing data shreds from any 32 of a set's 64, and decodes each batch into entries and legacy, v0 and v1 transactions.

I wrote it to read the shreds I record for trading research, and to test my reading of the format against real bytes before building anything on it.

It is an offline library and command line tool. It does not receive packets, know the leader schedule, verify proof of history, or execute transactions.

## How it works

A shred, byte by byte:

| Bytes | Data shred (1,203 bytes) | Code shred (1,228 bytes) |
|---|---|---|
| 0-63 | leader's signature over the set's Merkle root | same |
| 64 | variant `0x9_` (`0xB_` if resigned); low nibble = proof size | `0x6_` (`0x7_` if resigned) |
| 65-82 | slot u64, index u32, shred version u16, FEC set index u32 | same |
| 83-87 / 83-88 | parent offset u16, flags u8, size u16 | num data u16, num coding u16, position u16 |
| next | data `b[88:size]`, room for 963 bytes (899 if resigned) | Reed-Solomon parity, 987 bytes (923 if resigned) |
| next 32 | chained Merkle root (the previous set's root) | same |
| next 20 per level | Merkle proof (6 levels for a 32 + 32 set) | same |
| next 64 | retransmitter signature (resigned shreds only) | same |

Every offset is cited by file and line to Agave v4.4.0 in [`shredkit/layout.py`](shredkit/layout.py), so each one can be checked against the source.

The modules follow the path of the bytes: `layout` (headers and offsets), `merkle` (tree, proofs, signature), `rs` (Reed-Solomon), `fec` (recovering a set), `entries` (deshredding a batch), `tx` (transactions), `rawrec` (reading a recording), plus `build` and `synth` for signed synthetic data.

The design choices, and why each is there:

- **One shred is enough to authenticate it.** The leaf hash covers byte 64 up to the end of the chained root, the proof walks it up to the root, and the leader's Ed25519 signature is over that root. `verify_leader_signature` needs only the shred and the leader's public key. A test flips one bit in every signed byte of a data shred and of a code shred: every change is rejected. A second test checks the opposite: the retransmitter signature, which the leader does not sign, can change without affecting the result.
- **Malformed is not forged.** A shred that cannot be parsed, or whose proof cannot be walked, raises `ShredError`. A well-formed shred with a bad signature returns `False`. Counting the two apart separates a parser problem from a wrong key or a forged shred.
- **Reed-Solomon that matches the crate Agave uses** (`reed-solomon-erasure` 6.0.0): GF(2^8) with polynomial 29, and a coding matrix built from a Vandermonde matrix V as V * inverse(V[:32]). The first 32 rows are the identity, so data shards pass through unchanged and any 32 of the 64 shards rebuild them. numpy does the byte arithmetic (one lookup in a 256 x 256 product table, then an XOR); the 32 x 32 inversions are plain Python.
- **Recovery is checked against the signed root.** Reed-Solomon decodes wrong input into wrong data without complaint. So after decoding, `recover_set` re-encodes the parity, rebuilds every shred of the set byte for byte (headers, chained root, proofs) and requires the new Merkle root to equal the root the received shreds prove. Recovered shreds then pass the same signature check as received ones. A test corrupts one byte of the decoder's output, and recovery must refuse it.
- **Deshredding follows Agave's rules**: consecutive indexes, and the DATA_COMPLETE flag on the last shred of the batch and nowhere before it. The bytes are then decoded as `Vec<Entry>`, where each entry is `num_hashes`, `hash` and its transactions. What follows the entries is classified, not ignored: nothing (`exact`), only zeros (`zero_padded`, accepted), or anything else (`trailing_bytes`, reported as a failure). The Validation section explains why zero padding is accepted.
- **The transaction format is decided by the first byte.** A legacy or v0 transaction starts with its signature count, one byte below 128. A v1 transaction (SIMD-0385) starts with `0x81` and puts its signatures at the end. Reading v1 signature-first does not raise an error; it returns the wrong signature, so the format is settled before anything else is read, and a test checks the misreading.
- **Every read is bounds-checked.** All decoders read through one cursor that raises on a short read, and entry and transaction counts are checked against the bytes left before any loop, so a bad length field cannot run the parser past the end or hand back a short slice.

### Reading a rawrec recording

rawrec keeps each data shred's bytes (`b[88:size]`) at first arrival in `pay_*.bin`, and the header bytes, flags included, in `rec_*.bin`. `shredkit rawrec` reads both: the DATA_COMPLETE flags from the record files, which mark where each batch ends, then the payloads slot by slot, keeping about `--window` slots in memory. It parses both file formats itself, so shredkit does not depend on rawrec's code.

## Usage

Python 3.11 or newer, with numpy and cryptography:

```
pip install .
pip install ".[test]" && pytest        # 150 tests, synthetic data only, a few seconds
```

`shredkit demo` builds a signed synthetic slot (3 batches in 4 erasure sets, legacy, v0 and v1 transactions, one batch zero-padded), checks the signatures, drops 32 of the 64 shreds in every set, recovers, deshreds, and compares everything with what it built:

```
$ shredkit demo
built slot 1000: 3 batches, 4 erasure sets, 256 shreds (128 data, 128 code)
leader signature: 256 / 256 verify; after one random bit flip each: 0; with another key: 0
dropped 32 of 64 shreds in each set, rebuilt 68 data shreds; identical to the originals: True
  batch shreds 0-63: 31 entries, 120 txs {'legacy': 40, 'v0': 40, 'v1': 40}, exact
  batch shreds 64-95: 6 entries, 25 txs {'legacy': 9, 'v0': 8, 'v1': 8}, zero_padded, 700 zero bytes after the entries
  batch shreds 96-127: 8 entries, 42 txs {'legacy': 14, 'v0': 14, 'v1': 14}, exact
transactions: 187 parsed, identical to the ones built: True, signatures verify: True
all checks passed
```

The same steps from Python, on a synthetic slot standing in for received packets:

```python
import random

from shredkit import batch_ranges, deshred, group_by_set, parse_shred, recover_set, verify_leader_signature
from shredkit.synth import make_key, synthetic_slot
from shredkit.tx import public_key_bytes

rng = random.Random(1)
leader = make_key(rng)
received = synthetic_slot(rng, leader, 1000).payloads    # stand-in for one slot's UDP payloads

shreds = [parse_shred(p) for p in received]
assert all(verify_leader_signature(s, public_key_bytes(leader)) for s in shreds)

data = {}
for members in group_by_set(shreds).values():              # {(slot, fec_set_index): [shreds]}
    survivors = rng.sample(members, 32)                    # lose any half of every set
    data.update((s.index, s) for s in recover_set(survivors).data_shreds)

ends = [i for i, s in data.items() if s.data_complete]
complete, incomplete = batch_ranges(data, ends)
for span in complete:
    batch = deshred([data[i] for i in span])
    print(span, batch.status, len(batch.entries), "entries", len(batch.transactions), "transactions")
```

```
range(0, 64) exact 31 entries 120 transactions
range(64, 96) exact 6 entries 25 transactions
range(96, 128) exact 8 entries 42 transactions
```

A rawrec run directory (`synth-run` writes a synthetic one in the same format):

```
$ shredkit synth-run demo_run --slots 2
$ shredkit rawrec demo_run
slot 1000: 128 data shreds, 3 complete batches {'exact': 2, 'zero_padded': 1}, 0 incomplete, txs {'legacy': 63, 'v0': 62, 'v1': 62}
slot 1001: 128 data shreds, 3 complete batches {'exact': 2, 'zero_padded': 1}, 0 incomplete, txs {'legacy': 66, 'v0': 65, 'v1': 64}
total: 2 slots, 6 complete batches {'exact': 4, 'zero_padded': 2}, 0 incomplete, txs {'legacy': 129, 'v0': 127, 'v1': 126}; payload records 256, duplicates 0 (conflicting 0), late 0, torn bytes 0
```

`--json` prints one object per batch instead; `--txs` also lists every transaction's first signature in base58. The exit code is 0 when every complete batch decodes, 1 if any batch fails or has non-zero bytes after its entries, and 2 when the input cannot be read.

## Validation on real shreds

Before building on this code I wrote six pass/fail checks, then ran them on one real capture of Solana shred traffic: 9,223 shreds (4,615 data, 4,608 code). These are the results of that one run on that one capture. The capture is not in this repository. The checks ran on an earlier single-file version of this code; this package is a rewrite of it, and on synthetic batches the two produce identical shreds, Merkle roots, parity, recovered shards and transaction parses. The rewrite adds header range checks that were not part of that run.

| Check | Result |
|---|---|
| Lengths and variants | all 9,223 are 1,203 bytes (data) or 1,228 bytes (code), variants `0x96` / `0x66` and resigned `0xB6` / `0x76` |
| One root per erasure set | 145 of 145 sets with 2 or more shreds have one Merkle root and one signature |
| Leader signature | 3,200 of 3,200 shreds from the slots whose leader I had on file verify; with one bit flipped, 0 of 3,200 do |
| Reed-Solomon encode | on all 144 complete sets, parity computed from the 32 data shreds equals the 32 real coding shreds byte for byte |
| Reed-Solomon recover | on all 144, the 32 coding shreds alone rebuild the 32 data shreds exactly |
| Transactions | first signature correct on 120 of 120 transactions (60 v1, 30 v0, 30 legacy) taken from archived blocks, not from this capture; Ed25519 verifies over the message on all 120; the earlier signature-first reading of v1 matched 0 of 60 |
| **Deshred: failed** | of 95 complete batches, 64 decoded as `Vec<Entry>` ending exactly at the batch's last byte; 31 did not |

The deshred check failed as written. After seeing the failure I looked at the bytes that followed the entries in the 31 batches; this is a check made after the fact, so it carries less weight than the six above. They were zeros in all 31. Every batch in one leader window was exact and every batch in another was padded, which points to the producing validator rather than damage in transit; I have not confirmed which validator software does it. All 95 batches decoded to 7,957 transactions (5,710 legacy, 2,247 v0; the capture held no v1).

The rule "decode to the last byte" was stricter than the network. Agave's decoder does not require the entries to fill the batch: its own `Shredder::deshred` returns a zero-filled buffer for an empty batch and relies on it decoding as no entries (`solana-ledger` 4.4.1, `shredder.rs:233-242`). shredkit now accepts a zero tail and reports it as `zero_padded`, and still fails a non-zero one. That change is tested on synthetic padded batches only; it has not been run on the real capture.

## Limitations

- Speed. Pure Python with numpy; on one laptop core, about 120 us to verify a shred (mostly the Ed25519 check), 22 ms to recover a set that lost all 32 data shreds, and 18 us to decode a transaction. That suits offline analysis of a capture. It cannot keep up with mainnet in real time.
- Only chained Merkle shreds (variants `0x6_`, `0x7_`, `0x9_`, `0xB_`), the format Agave v4.4.0 sends. Legacy and unchained Merkle shreds are rejected with `ShredError`.
- The synthetic builder fills whole 32-shred sets and pads the last set with short or empty data shreds. It does not copy how Agave sizes a short last set. It is a test fixture, not a shredder.
- Recovery reads a set's size from its code shred headers. With no code shred received the set is assumed to be 32 + 32, which only matters when nothing is missing.
- The retransmitter signature of resigned shreds is not verified, and recovery cannot rebuild it (recovered shreds carry zeros there).
- Entry hashes and `num_hashes` are decoded, not checked against proof of history. A batch whose entry count is 0 and which carries more data is reported as a block marker with its version, not decoded. That path, and v1 transactions inside shreds, are tested on synthetic data only.
- Transactions are parsed and their signatures checked. They are not sanitized as the runtime does (account index ranges, duplicate keys, address lookup tables).
- From a rawrec recording: payloads carry no signatures or proofs, and rawrec does not keep coding shreds' payloads, so batches from a recording are neither signature-checked nor recoverable. If the record of a DATA_COMPLETE shred was lost, two batches merge and show up as `trailing_bytes`. A payload arriving more than `--window` slots late is counted and skipped.
- No leader schedule. You supply the leader's public key.

## Authorship

Designed by me; implemented with AI coding agents under my design and review; I can walk through every part of it.

## License

MIT. See [LICENSE](LICENSE).
