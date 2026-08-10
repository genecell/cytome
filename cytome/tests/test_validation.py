from __future__ import annotations

import cytome


class TestValidation:
    def test_valid_dataset(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        report = ds.validate()
        assert report.passed
        ds.close()

    def test_detect_missing_table(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        ds._conn.execute("DROP TABLE samples")
        report = ds.validate()
        assert not report.passed
        assert any("missing_tables" in m for m in report.checks_failed)
        ds.close()

    def test_repair_orphans(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        ds._conn.execute(
            """
            INSERT INTO matrix_chunks(
                matrix_name, chunk_idx, row_start, row_end, n_nonzero,
                data_blob, indices_blob, indptr_blob, dtype, compression
            ) VALUES ('orphan', 0, 0, 0, 0, x'', x'', x'', 'float32', 'zlib')
            """
        )
        ds.repair()
        count = ds._conn.execute(
            "SELECT COUNT(*) FROM matrix_chunks WHERE matrix_name='orphan'"
        ).fetchone()[0]
        assert count == 0
        ds.close()
