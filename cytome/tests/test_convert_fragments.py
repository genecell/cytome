from __future__ import annotations

import gzip

from cytome.io.convert_fragments import export_fragments, import_fragments
from cytome.io.sqlite_engine import create_database, close_database


class TestConvertFragments:
    def test_import_fragments(self, tmp_cytome, synthetic_fragments_file):
        conn = create_database(tmp_cytome)
        mapping = {f"ACGT{i:04d}-1": i for i in range(500)}
        import_fragments(conn, synthetic_fragments_file, mapping, build_index=True)
        n = conn.execute("SELECT SUM(n_fragments) FROM fragment_meta").fetchone()[0]
        assert int(n) > 0
        close_database(conn)

    def test_export_fragments(self, tmp_cytome, synthetic_fragments_file, tmp_path):
        conn = create_database(tmp_cytome)
        mapping = {f"ACGT{i:04d}-1": i for i in range(500)}
        import_fragments(conn, synthetic_fragments_file, mapping, build_index=True)
        rev = {i: f"ACGT{i:04d}-1" for i in range(500)}
        out = tmp_path / "export.fragments.tsv.gz"
        export_fragments(conn, out, rev)
        assert out.exists()
        with gzip.open(out, "rt") as f:
            first = f.readline().strip().split("\t")
        assert len(first) == 5
        close_database(conn)
