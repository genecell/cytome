from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import cytome


def _mk_ds(path: Path):
    ds = cytome.create(path)
    ds.set_entity("cells", {"barcode": [f"c{i}" for i in range(20)], "cell_type": ["A"] * 10 + ["B"] * 10})
    ds.set_entity("genes", {"gene_id": [f"g{i}" for i in range(8)], "symbol": [f"g{i}" for i in range(8)]})
    ds.add_matrix("RNA_counts", sp.random(20, 8, density=0.3, format="csr", dtype=np.float32, random_state=4))
    ds.flush()
    ds.close()


def _run(*args):
    return subprocess.run([sys.executable, "-m", "cytome.cli.main", *args], capture_output=True, text=True)


def test_no_args_shows_help():
    p = _run()
    assert p.returncode != 0
    assert "usage" in p.stdout.lower() or "usage" in p.stderr.lower()


def test_info_json(tmp_path):
    p = tmp_path / "a.cytome"
    _mk_ds(p)
    r = _run("info", str(p), "--json")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["n_cells"] == 20


def test_validate_cli(tmp_path):
    p = tmp_path / "a.cytome"
    _mk_ds(p)
    r = _run("validate", str(p))
    assert r.returncode == 0
    assert "PASSED" in r.stdout


def test_merge_subset_downsample_cli(tmp_path):
    p1, p2, pm = tmp_path / "s1.cytome", tmp_path / "s2.cytome", tmp_path / "m.cytome"
    _mk_ds(p1)
    _mk_ds(p2)
    r = _run("merge", str(p1), str(p2), "-o", str(pm))
    assert r.returncode == 0
    sub = tmp_path / "sub.cytome"
    r = _run("subset", str(pm), "-o", str(sub), "--query", "cell_type == 'A'")
    assert r.returncode == 0
    small = tmp_path / "small.cytome"
    r = _run("downsample", str(pm), "-o", str(small), "--n-cells", "10")
    assert r.returncode == 0


def test_invalid_command_exits_nonzero(tmp_path):
    r = _run("unknown")
    assert r.returncode != 0
