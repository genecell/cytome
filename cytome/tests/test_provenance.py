from __future__ import annotations

import cytome


class TestProvenance:
    def test_log_operation(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        pid = ds.provenance.log(
            operation="test",
            function_name="fn",
            parameters={"a": 1},
            package_name="cytome",
            package_version="0.1.0",
            input_objects=["x"],
            output_objects=["y"],
        )
        assert pid > 0
        ds.flush()
        ds.close()

    def test_show(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        ds.provenance.log(
            operation="test",
            function_name="fn",
            parameters={"a": 1},
            package_name="cytome",
            package_version="0.1.0",
            input_objects=["x"],
            output_objects=["y"],
        )
        text = ds.provenance.show()
        assert "fn" in text
        ds.close()

    def test_get_for_object(self, tmp_cytome):
        ds = cytome.create(tmp_cytome)
        ds.provenance.log(
            operation="test",
            function_name="fn",
            parameters={"a": 1},
            package_name="cytome",
            package_version="0.1.0",
            input_objects=["RNA_counts"],
            output_objects=["RNA_norm"],
        )
        rows = ds.provenance.get_for_object("RNA_counts")
        assert len(rows) >= 1
        ds.close()
