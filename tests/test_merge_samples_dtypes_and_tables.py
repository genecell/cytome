"""merge keeps cells' samples, writes one dtype per matrix, and carries
embeddings, raw, records, annotations, images and fragments.

Audit 2026-10-27, part 4 critical 4 (inputs of different dtypes: a count of 2
read back as 1,073,741,824), major 9 (the default batch_key replaced each
cell's sample_id, and the samples table came out empty), major 26 (only the
last input's embedding survived, a raw matrix wider than its features raised or
was written under the wrong width, obs_columns='shared' raised, the
``_anndata_*`` records were prefixed and lost, gene models, var embeddings and
spatial data were not carried), improvement 4 (a streaming merge, fragments
merged a chunk of each input at a time) and the second look's merge design
(barcodes kept, sample_id filled only when empty, the samples table filled,
repeated barcodes reported).
"""
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cytome
import cytome.io.merge as merge_mod
from cytome.io.convert_fragments import _update_fragment_meta, _write_block_chunks
from cytome.io.compression import decompress_blob

anndata = pytest.importorskip("anndata")


def _store(path, n, dtype=np.float32, sample_id=None, value=2, emb=None, prefix="AAA",
           frags=None, genes=("g0", "g1")):
    ds = cytome.create(str(path))
    cells = pd.DataFrame({"barcode": [f"{prefix}{i}" for i in range(n)],
                          "ct": pd.Categorical(["T", "B"] * (n // 2) + ["T"] * (n % 2))})
    if sample_id is not None:
        cells["sample_id"] = sample_id
    ds.set_entity("cells", cells)
    ds.set_entity("genes", pd.DataFrame({"gene_id": list(genes)}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.full((n, len(genes)), value, dtype=dtype)))
    if emb is not None:
        ds.add_embedding("RNA_umap", np.full((n, emb), float(n)), flush=False)
    ds.flush()
    from cytome.core.column_meta import store_column_meta
    store_column_meta(ds._conn, "cells", cells)
    ds._conn.commit()
    if frags is not None:
        s, e, c, chunk = frags
        _write_block_chunks(ds._conn, "chr1", np.asarray(s, np.int32), np.asarray(e, np.int32),
                            np.asarray(c, np.int32), chunk_size=chunk, compression="lz4",
                            compression_level=1, chunk_idx_start=0, encoding=1)
        _update_fragment_meta(ds._conn, "chr1", len(s))
        ds._conn.commit()
    ds.close()
    return str(path)


def _merge(paths, out, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cytome.merge(paths, str(out), **kw)


def test_inputs_of_different_dtypes_merge_in_one_dtype(tmp_path):
    a = _store(tmp_path / "a.cytome", 3, np.int32)
    b = _store(tmp_path / "b.cytome", 2, np.float32)
    m = _merge([a, b], tmp_path / "m.cytome")
    assert m.RNA.counts.dtype == np.float64
    assert {r[0] for r in m._conn.execute("SELECT DISTINCT dtype FROM matrix_chunks")} == {"float64"}
    assert m.RNA.counts.to_memory().toarray()[:, 0].tolist() == [2, 2, 2, 2, 2]
    m.close()
    m = _merge([a, b], tmp_path / "f.cytome", dtype="float32")
    assert m.RNA.counts.dtype == np.float32
    m.close()
    c = _store(tmp_path / "c.cytome", 2, np.float32, value=0.5)
    with pytest.raises(ValueError, match="do not fit"):
        _merge([a, c], tmp_path / "i.cytome", dtype="int32")
    assert not os.path.exists(tmp_path / "i.cytome")
    assert not os.path.exists(str(tmp_path / "i.cytome") + ".partial")


def test_cells_keep_their_samples_and_the_samples_table_is_filled(tmp_path):
    a = _store(tmp_path / "a.cytome", 3, sample_id=["donorX", "donorY", "donorX"])
    b = _store(tmp_path / "b.cytome", 2)
    m = _merge([a, b], tmp_path / "m.cytome")
    assert m.cells["sample_id"].tolist() == ["donorX", "donorY", "donorX", "b", "b"]
    assert m.cells["batch"].tolist() == ["a", "a", "a", "b", "b"]
    samples = dict(m._conn.execute("SELECT sample_id, n_cells FROM samples"))
    assert samples == {"donorX": 2, "donorY": 1, "b": 2}
    assert dict(m._conn.execute("SELECT sample_id, batch FROM samples"))["donorY"] == "a"
    assert m.validate(integrity=False).passed
    m.close()
    m = _merge([a, b], tmp_path / "s.cytome", batch_key="sample_id")
    assert m.cells["sample_id"].tolist() == ["a", "a", "a", "b", "b"]
    m.close()


def test_repeated_barcodes_are_kept_and_reported(tmp_path):
    a = _store(tmp_path / "a.cytome", 3)
    b = _store(tmp_path / "b.cytome", 2)
    with pytest.warns(UserWarning, match="2 barcodes occur in more than one input"):
        m = cytome.merge([a, b], str(tmp_path / "m.cytome"))
    assert m.cells["barcode"].tolist() == ["AAA0", "AAA1", "AAA2", "AAA0", "AAA1"]
    m.close()


def test_embeddings_are_concatenated(tmp_path):
    a = _store(tmp_path / "a.cytome", 3, emb=2)
    b = _store(tmp_path / "b.cytome", 2, emb=2)
    c = _store(tmp_path / "c.cytome", 1)
    m = _merge([a, b, c], tmp_path / "m.cytome", include_embeddings=True)
    e = m.embeddings["RNA_umap"]
    assert e.shape == (6, 2)
    assert e[:3].tolist() == [[3, 3]] * 3 and e[3:5].tolist() == [[2, 2]] * 2
    assert np.isnan(e[5]).all()
    m.close()


def test_obs_columns_shared_and_categories(tmp_path):
    a = _store(tmp_path / "a.cytome", 3)
    b = _store(tmp_path / "b.cytome", 2)
    m = _merge([a, b], tmp_path / "m.cytome", obs_columns="shared")
    assert set(m.cells.columns) >= {"barcode", "sample_id", "batch", "ct"}
    assert str(m.to_anndata().obs["ct"].dtype) == "category"
    m.close()


def test_raw_wider_than_x_and_conversion_records(tmp_path):
    def conv(path, genes, raw_genes, n=3):
        full = anndata.AnnData(X=sp.csr_matrix(np.arange(n * len(raw_genes), dtype=np.float32)
                                               .reshape(n, len(raw_genes))),
                               var=pd.DataFrame(index=raw_genes),
                               obs=pd.DataFrame(index=[f"{path.stem}{i}" for i in range(n)]))
        a = full[:, list(genes)].copy()
        a.raw = full
        a.layers["norm"] = a.X * 0.5
        cytome.from_anndata(a, output=str(path)).close()
        return str(path)
    a = conv(tmp_path / "a.cytome", ["g0", "g1"], ["g0", "g1", "g2", "g3"])
    b = conv(tmp_path / "b.cytome", ["g1", "g2"], ["g1", "g2", "g4"])
    m = _merge([a, b], tmp_path / "m.cytome")
    raw = m.RNA.layer("raw_X")
    assert raw.shape == (6, 5)                                    # g0..g4
    names = list(pd.read_sql_query("SELECT var_name FROM _raw_var ORDER BY var_idx",
                                   m._conn)["var_name"])
    assert names == ["g0", "g1", "g2", "g3", "g4"]
    full = raw.to_memory().toarray()
    assert full[0, :4].tolist() == [0, 1, 2, 3] and full[3, [1, 2, 4]].tolist() == [0, 1, 2]
    assert m.metadata["_anndata_X_layer"] == "RNA_counts"
    ad = m.to_anndata()
    assert "norm" in ad.layers and ad.raw is not None and ad.raw.shape == (6, 5)
    m.close()


def test_gene_models_var_embeddings_and_spatial_are_carried(tmp_path):
    paths = []
    for name, n in (("a", 3), ("b", 2)):
        p = _store(tmp_path / f"{name}.cytome", n)
        ds = cytome.open(p)
        ds._conn.execute("INSERT INTO _gene_annotation VALUES ('g0','chr1',0,100,'+','G0',"
                         "'protein_coding','test')")
        ds.add_var_embedding("RNA_varm_pc", np.ones((2, 3)))
        ds.add_spatial_image("lib", "hires", np.full((2, 2), n, dtype=np.uint8),
                             scalefactors={"tissue_hires_scalef": 0.5})
        ds.set_spatial_coords(np.arange(n * 2, dtype=float).reshape(n, 2))
        ds.close()
        paths.append(p)
    m = _merge(paths, tmp_path / "m.cytome")
    assert m.gene_annotation_info()["n_genes"] == 1
    assert m.var_embeddings.keys() == ["RNA_varm_pc"]
    assert sorted(m.spatial_images.keys()) == [("b:lib", "hires"), ("lib", "hires")]
    assert m.cells_in_region((0, 10), (0, 10)).tolist() == [0, 1, 2, 3, 4]
    m.close()


def test_fragments_are_merged_by_start_whatever_the_cell_idx(tmp_path):
    rng = np.random.default_rng(3)

    def frags(n_cells, n, seed):
        r = np.random.default_rng(seed)
        s = np.sort(r.integers(0, 10_000, n)).astype(np.int32)
        return s, s + 50, r.integers(0, n_cells, n), 7           # chunks of 7 fragments
    fa, fb = frags(3, 40, 1), frags(2, 25, 2)
    a = _store(tmp_path / "a.cytome", 3, frags=fa)
    b = _store(tmp_path / "b.cytome", 2, frags=fb)
    ds = cytome.open(b)                                          # B's keys are not its rows
    ds._conn.execute("PRAGMA foreign_keys=OFF")
    ds._conn.execute("UPDATE cells SET cell_idx = cell_idx + 100")
    ds._conn.commit()
    # its fragments name those keys
    ds._conn.execute("DELETE FROM fragment_chunks")
    _write_block_chunks(ds._conn, "chr1", fb[0], fb[1], (fb[2] + 100).astype(np.int32),
                        chunk_size=7, compression="lz4", compression_level=1,
                        chunk_idx_start=0, encoding=1)
    ds._conn.commit()
    ds.close()
    m = _merge([a, b], tmp_path / "m.cytome")
    got = [np.concatenate(x) for x in zip(*m.fragments.iter_chromosome_chunks("chr1"))]
    s, e, c = got
    assert np.all(np.diff(s) >= 0)
    want = sorted(list(zip(fa[0], fa[2])) + list(zip(fb[0], fb[2] + 3)))
    assert sorted(zip(s.tolist(), c.tolist())) == [(int(x), int(y)) for x, y in want]
    assert m.validate(integrity=False).passed
    m.close()


def test_k_way_merge_matches_a_sort(tmp_path):
    """The window logic of the chunk-at-a-time merge, against a full sort."""
    rng = np.random.default_rng(5)
    streams_data = []
    for k in range(4):
        s = np.sort(rng.integers(0, 2_000, 300)).astype(np.int32)
        streams_data.append((s, s + 5, np.full(300, k, dtype=np.int32)))

    def chunked(s, e, c, size):
        for i in range(0, s.size, size):
            yield s[i:i + size], e[i:i + size], c[i:i + size]
    identity = np.arange(4, dtype=np.int64)
    streams = [merge_mod._SortedStream(chunked(*d, size=int(rng.integers(5, 60))), identity)
               for d in streams_data]
    got = []

    class W:
        def add(self, s, e, c):
            got.append(s)
    assert merge_mod._kway(streams, W())
    s = np.concatenate(got)
    assert s.size == 1200 and np.all(np.diff(s) >= 0)
    assert np.array_equal(s, np.sort(np.concatenate([d[0] for d in streams_data])))


def test_an_existing_output_is_refused_and_a_failure_leaves_none(tmp_path):
    a = _store(tmp_path / "a.cytome", 3)
    b = _store(tmp_path / "b.cytome", 2)
    (tmp_path / "m.cytome").write_text("x")
    with pytest.raises(FileExistsError):
        _merge([a, b], tmp_path / "m.cytome")
    assert (tmp_path / "m.cytome").read_text() == "x"
