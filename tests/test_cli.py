import json
import random

from shredkit.build import build_batch
from shredkit.cli import main
from shredkit.rawrec import write_run
from shredkit.synth import make_key, random_batch


def test_demo(capsys):
    assert main(["demo", "--seed", "2", "--drop", "20"]) == 0
    assert "all checks passed" in capsys.readouterr().out


def test_synth_run_then_rawrec(tmp_path, capsys):
    assert main(["synth-run", str(tmp_path), "--slots", "2"]) == 0
    assert main(["rawrec", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "total: 2 slots, 6 complete batches" in out and "'zero_padded': 2" in out


def test_rawrec_json_lines(tmp_path, capsys):
    main(["synth-run", str(tmp_path), "--slots", "1"])
    capsys.readouterr()
    assert main(["rawrec", str(tmp_path), "--json"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["status"] for r in rows] == ["exact", "zero_padded", "exact"]


def test_rawrec_exit_code_flags_trailing_bytes(tmp_path, capsys):
    """Planted error: a batch with a non-zero byte after its entries must make the command fail."""
    rng = random.Random(3)
    blob, _ = random_batch(rng, n_entries=3, txs_per_entry=2)
    built = build_batch(make_key(rng), 9_000, blob + b"\x00\x00\x07")
    write_run(tmp_path, built.shreds)
    assert main(["rawrec", str(tmp_path)]) == 1
    assert "trailing_bytes" in capsys.readouterr().out


def test_rawrec_on_an_empty_directory(tmp_path, capsys):
    assert main(["rawrec", str(tmp_path)]) == 2
    assert "needs pay_" in capsys.readouterr().err
