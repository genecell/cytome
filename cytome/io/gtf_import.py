"""GTF → cytome annotation tables (cytome 0.2.2+; one reader for every GTF convention since 0.3.7).

Parses a GTF file (gzipped or not), converts 1-based closed coordinates to 0-based half-open
(BED convention), and writes rows to the ``_gene_annotation`` and ``_exon_annotation`` tables,
queryable via ``ds.query_gene_annotation()`` / ``ds.query_exon_annotation()``.

GTF files differ in what they spell out, and a reader that assumes one convention silently loses
genes on another:

- GENCODE and Ensembl write a ``gene`` line for every gene and name it in ``gene_name``
  (Ensembl leaves many non-model-species genes unnamed).
- NCBI RefSeq names genes in ``gene``; its ``gene_id`` is the symbol, suffixed ``_1``, ``_2`` where
  a symbol repeats. Files built from it can carry ``gene`` lines for some genes only (the Cell
  Ranger ARC Mmul_10 reference has them for chr1 alone, 2,799 of 34,643 genes).
- GTF 2.2 itself defines no ``gene`` or ``transcript`` lines: a gene is the set of rows sharing a
  ``gene_id`` (gffutils infers gene extents the same way).
- Ensembl names chromosomes ``1``, ``X``, ``MT``; UCSC-style stores use ``chr1``, ``chrX``, ``chrM``.

:func:`iter_gtf` reads any of them into one form, used by :func:`import_gtf` and
:func:`read_gtf` (and through it by PIASO's gene tracks), so every consumer sees the same genes:

- a gene's name is ``gene_name``, else ``gene``, else ``Name``, and is left empty otherwise (an ID
  is never stored as a name; look genes up by ID with ``gene_ids=``);
- a gene without a ``gene`` line gets one, spanning its transcripts (or, without transcript rows,
  its other rows); a ``gene`` line wins when there is one;
- the gene type is ``gene_type`` / ``gene_biotype`` from any of the gene's rows;
- the filters act on these genes and on transcripts as a whole, so they no longer depend on
  ``gene`` lines or on line order; genes a ``gene_biotypes`` filter drops for lack of any biotype
  are counted;
- chromosome names can be mapped to a store's own spelling (``chrom_map``; :func:`import_gtf`
  does it by default when the store holds fragments, peaks or tiles).
"""
from __future__ import annotations

import gzip
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

#: Version of the reader written to ``ds.metadata['gene_annotation']`` by :func:`import_gtf`.
#: Annotations imported before v2 (no ``reader_version``) took genes from ``gene`` lines only;
#: v3 infers ``gene_type`` from ``CDS`` / ``transcript_type`` / ``gbkey`` when ``gene`` lines are absent.
READER_VERSION = 2

DEFAULT_FEATURES = (
    "gene", "exon", "transcript",
    "CDS", "UTR",
    "five_prime_utr", "three_prime_utr",
)
GENE_COLUMNS = ["gene_id", "chrom", "start", "end", "strand", "gene_name", "gene_type", "source"]
FEATURE_COLUMNS = ["gene_id", "transcript_id", "exon_number", "chrom", "start", "end", "strand",
                   "feature", "transcript_type"]

# Pre-compiled regex for GTF attribute fields: ``key "value"``
_ATTR_PATTERN = re.compile(r'(\w+)\s+"([^"]*)"')


def _parse_attributes(attr_field: str) -> dict:
    """Parse a GTF attribute field into a dict.

    GTF attributes look like:
        gene_id "ENSMUSG00000031328"; gene_name "Dlx1"; ...

    Returns single-valued attrs only (last write wins for duplicates).
    Use :func:`_parse_attribute_tags` to extract all values of a
    multi-valued attribute like ``tag``.
    """
    return dict(_ATTR_PATTERN.findall(attr_field))


def _parse_attribute_tags(attr_field: str, key: str = "tag") -> list:
    """Extract all values for a multi-valued GTF attribute.

    Ensembl / GENCODE put one ``tag "X";`` entry per attribute value
    (e.g. ``tag "basic"; tag "Ensembl_canonical";``).
    ``_parse_attributes`` would silently drop all but the last —
    use this when checking for the presence of a specific tag.
    """
    return [v for (k, v) in _ATTR_PATTERN.findall(attr_field) if k == key]


def _gene_name_of(attrs: dict) -> Optional[str]:
    return attrs.get("gene_name") or attrs.get("gene") or attrs.get("Name") or None


def _gene_type_of(attrs: dict) -> Optional[str]:
    return attrs.get("gene_type") or attrs.get("gene_biotype") or None


# ---------------------------------------------------------------------------
# Chromosome names
# ---------------------------------------------------------------------------

def chrom_alternatives(name: str) -> list:
    """The other spellings of a chromosome name: ``8`` <-> ``chr8``, ``MT`` <-> ``chrM``."""
    if name.startswith("chr"):
        base = name[3:]
        return ["MT", "M"] if base in ("M", "MT") else [base]
    if name in ("MT", "M"):
        return ["chrM", "chrMT"]
    return ["chr" + name]


def harmonize_chrom(name: str, target_names) -> str:
    """``name`` in the spelling of ``target_names``: unchanged when it is already one of them or
    when no other spelling is; otherwise the first other spelling that is."""
    if name in target_names:
        return name
    for alt in chrom_alternatives(name):
        if alt in target_names:
            return alt
    return name


class _ChromMap:
    """Lazy per-name mapping to a set of target names; records what it renamed."""

    def __init__(self, target_names):
        self.target = set(target_names)
        self.cache: dict = {}
        self.renamed: dict = {}

    def __call__(self, name: str) -> str:
        out = self.cache.get(name)
        if out is None:
            out = harmonize_chrom(name, self.target)
            self.cache[name] = out
            if out != name:
                self.renamed[name] = out
        return out


def store_chromosomes(ds) -> set:
    """Chromosome names of a store's genomic data: fragments, peaks and tiles (empty if none)."""
    names: set = set()
    for sql in ("SELECT chrom FROM fragment_meta WHERE n_fragments > 0",
                "SELECT DISTINCT chrom FROM fragment_chunks",
                "SELECT DISTINCT chr FROM peaks",
                "SELECT DISTINCT chr FROM tiles"):
        try:
            names.update(str(r[0]) for r in ds._conn.execute(sql).fetchall() if r[0] is not None)
        except Exception:     # table absent in this store
            continue
    return names


def _resolve_chrom_map(chrom_map):
    """``None`` / dict / callable / set of target names -> callable or None."""
    if chrom_map is None:
        return None
    if callable(chrom_map):
        return chrom_map
    if isinstance(chrom_map, dict):
        return lambda name: chrom_map.get(name, name)
    return _ChromMap(chrom_map)


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------

def _transcript_id_regex(transcript_id_pattern, transcript_id_prefixes):
    if transcript_id_pattern is not None and transcript_id_prefixes is not None:
        raise ValueError(
            "cytome.import_gtf: pass either transcript_id_pattern "
            "OR transcript_id_prefixes, not both."
        )
    if transcript_id_prefixes:
        return re.compile("^(" + "|".join(re.escape(p) for p in transcript_id_prefixes) + ")")
    if transcript_id_pattern is not None:
        return re.compile(transcript_id_pattern)
    return None


def _rows(gtf_path: Path):
    """``(chrom, feature, start0, end, strand, attr_field)`` for every data line."""
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            try:
                start = int(fields[3]) - 1      # GTF 1-based closed -> 0-based half-open
                end = int(fields[4])
            except ValueError:
                continue
            strand = fields[6] if fields[6] in ("+", "-", ".") else "."
            yield fields[0], fields[2], start, end, strand, fields[8]


class _Gene:
    """What the rows of one gene_id say about the gene."""

    __slots__ = ("line", "name", "type", "tx", "rows", "loci", "has_cds", "tx_types", "product")

    def __init__(self):
        self.line = None        # (chrom, start, end, strand) of its gene line, the longest if several
        self.name = None
        self.type = None
        self.tx = {}            # (chrom, strand) -> [min start, max end] over transcript rows
        self.rows = {}          # (chrom, strand) -> [min start, max end] over all other rows
        self.loci = Counter()   # (chrom, strand) -> number of non-gene rows
        self.has_cds = False
        self.tx_types = set()
        self.product = None

    def add(self, chrom, feature, start, end, strand, attrs):
        if feature == "gene":
            # the gene line's own name and type win over its rows', wherever it sits in the file
            self.name = _gene_name_of(attrs) or self.name
            self.type = _gene_type_of(attrs) or self.type
            if self.line is None or end - start > self.line[2] - self.line[1]:
                self.line = (chrom, start, end, strand)
            return
        if self.name is None:
            self.name = _gene_name_of(attrs)
        if self.type is None:
            self.type = _gene_type_of(attrs)
        if feature == "CDS":
            self.has_cds = True
        tt = (
            attrs.get("transcript_type")
            or attrs.get("transcript_biotype")
            or attrs.get("gbkey")
        )
        if tt and tt not in ("exon", "GF", "Gene"):
            self.tx_types.add(tt)
        if self.product is None:
            self.product = attrs.get("product") or attrs.get("description")
        key = (chrom, strand)
        self.loci[key] += 1
        span = self.rows.get(key)
        self.rows[key] = [start, end] if span is None else [min(span[0], start), max(span[1], end)]
        if feature == "transcript":
            span = self.tx.get(key)
            self.tx[key] = [start, end] if span is None else [min(span[0], start), max(span[1], end)]

    def inferred_type(self) -> Optional[str]:
        """Explicit ``gene_type`` / ``gene_biotype``, else inferred from CDS / transcript rows."""
        if self.type is not None:
            return self.type
        if self.has_cds or "mRNA" in self.tx_types or "protein_coding" in self.tx_types or "CDS" in self.tx_types:
            return "protein_coding"
        if "lnc_RNA" in self.tx_types or "lncRNA" in self.tx_types or "ncRNA" in self.tx_types:
            return "lncRNA"
        for bt in ("tRNA", "rRNA", "snRNA", "snoRNA", "miRNA", "scRNA", "misc_RNA"):
            if bt in self.tx_types:
                return bt
        if "pseudogene" in self.tx_types or "transcribed_pseudogene" in self.tx_types:
            return "pseudogene"
        if self.tx_types:
            return sorted(self.tx_types)[0]
        return None

    def locus(self):
        """(chrom, start, end, strand, inferred): the gene line, else its transcripts' span on the
        locus with most rows, else that locus's span over all rows."""
        if self.line is not None:
            return (*self.line, False)
        if not self.loci:
            return None
        key = max(self.loci, key=self.loci.get)        # ties: the locus met first
        span = self.tx.get(key) or self.rows[key]
        return (key[0], span[0], span[1], key[1], True)


def iter_gtf(
    gtf_path,
    *,
    feature_filter=DEFAULT_FEATURES,
    source_label: Optional[str] = None,
    transcript_id_pattern: Optional[str] = None,
    transcript_id_prefixes=None,
    transcript_tags=None,
    gene_biotypes=None,
    chrom_map=None,
    stats: Optional[dict] = None,
) -> Iterator[tuple]:
    """Read a GTF of any convention (module docstring); yield normalized rows.

    Yields ``("feature", row)`` for every kept non-gene row (``row`` in :data:`FEATURE_COLUMNS`
    order, 0-based half-open), then ``("gene", row)`` for every kept gene (:data:`GENE_COLUMNS`
    order) if ``'gene'`` is in ``feature_filter``. Gene rows are produced for every gene_id that
    has rows (its ``gene`` line, or one spanning its transcripts), whatever ``feature_filter``
    imports of its other rows, and are kept under the transcript filters as ``gene`` lines always
    were; ``gene_biotypes`` drops whole genes.

    ``chrom_map``: ``None`` (names as in the file), a dict, a callable, or a set of target names
    (each name is put in their spelling when its other spelling is one of them).
    ``stats``: a dict filled with counts (gene lines, inferred genes, named genes, rows per
    feature, rows and genes dropped by each filter, genes split across loci, chromosomes renamed).
    The file is read twice when ``gene_biotypes`` or ``transcript_tags`` is given (the decision
    for a row can depend on rows further down), once otherwise.
    """
    gtf_path = Path(gtf_path)
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")
    if source_label is None:
        source_label = gtf_path.name
        for suffix in (".gz", ".gtf"):
            if source_label.endswith(suffix):
                source_label = source_label[: -len(suffix)]
    tx_regex = _transcript_id_regex(transcript_id_pattern, transcript_id_prefixes)
    tags_set = set(transcript_tags) if transcript_tags else None
    biotypes_set = set(gene_biotypes) if gene_biotypes else None
    features = set(feature_filter)
    cmap = _resolve_chrom_map(chrom_map)
    st = stats if stats is not None else {}
    st.update(gene_lines=0, genes=0, genes_inferred=0, genes_named=0, features={},
              rows_dropped_transcript_id=0, rows_dropped_tag=0, rows_dropped_biotype=0,
              genes_dropped_biotype=0, genes_dropped_no_biotype=0, genes_split_across_loci=0,
              split_examples=[], chrom_renamed={}, source=source_label)

    # Pass 1, only for filters whose decision can depend on later rows: gene types (from any of a
    # gene's rows) and the transcripts that carry a requested tag (on any of their rows).
    gene_type: dict = {}
    allowed_tx = set() if tags_set is not None else None
    if biotypes_set is not None or tags_set is not None:
        for chrom, feature, start, end, strand, attr_field in _rows(gtf_path):
            attrs = _parse_attributes(attr_field)
            gid = attrs.get("gene_id")
            if not gid:
                continue
            if biotypes_set is not None and gid not in gene_type:
                t = _gene_type_of(attrs)
                if t is not None:
                    gene_type[gid] = t
            if tags_set is not None:
                tid = attrs.get("transcript_id")
                if tid and any(t in tags_set for t in _parse_attribute_tags(attr_field, "tag")):
                    allowed_tx.add(tid)

    genes: dict = {}
    feat_counts = Counter()
    for chrom, feature, start, end, strand, attr_field in _rows(gtf_path):
        attrs = _parse_attributes(attr_field)
        gid = attrs.get("gene_id")
        if not gid:
            continue
        if cmap is not None:
            chrom = cmap(chrom)
        g = genes.get(gid)
        if g is None:
            g = genes[gid] = _Gene()
        g.add(chrom, feature, start, end, strand, attrs)
        if feature == "gene":
            st["gene_lines"] += 1
            continue
        if feature not in features:
            continue
        if biotypes_set is not None and gene_type.get(gid) not in biotypes_set:
            st["rows_dropped_biotype"] += 1
            continue
        tid = attrs.get("transcript_id") or None
        if tx_regex is not None and tid and not tx_regex.match(tid):
            st["rows_dropped_transcript_id"] += 1
            continue
        if allowed_tx is not None and (tid is None or tid not in allowed_tx):
            st["rows_dropped_tag"] += 1
            continue
        exon_num = attrs.get("exon_number")
        try:
            exon_num = int(exon_num) if exon_num else None
        except ValueError:
            exon_num = None
        feat_counts[feature] += 1
        yield ("feature", (gid, tid, exon_num, chrom, start, end, strand, feature,
                           attrs.get("transcript_type") or attrs.get("transcript_biotype")))

    st["features"] = dict(feat_counts)
    if isinstance(cmap, _ChromMap):
        st["chrom_renamed"] = dict(cmap.renamed)
    if "gene" not in features:
        return
    for gid, g in genes.items():
        loc = g.locus()
        if loc is None:
            continue
        chrom, start, end, strand, inferred = loc
        gtype = g.inferred_type() if biotypes_set is None else gene_type.get(gid)
        if biotypes_set is not None and gtype not in biotypes_set:
            st["genes_dropped_biotype"] += 1
            if gtype is None:
                st["genes_dropped_no_biotype"] += 1
            continue
        if inferred and len(g.loci) > 1:
            st["genes_split_across_loci"] += 1
            if len(st["split_examples"]) < 5:
                st["split_examples"].append(gid)
        st["genes"] += 1
        st["genes_inferred"] += int(inferred)
        st["genes_named"] += int(g.name is not None)
        yield ("gene", (gid, chrom, start, end, strand, g.name, gtype, source_label))


def read_gtf(gtf_path, *, stats: Optional[dict] = None, **kwargs):
    """A GTF as two DataFrames, ``(genes, features)``, in the columns and 0-based half-open
    coordinates of ``ds.query_gene_annotation()`` / ``ds.query_exon_annotation()``.

    Same reader and keyword arguments as :func:`iter_gtf`; for callers without a cytome (e.g. a
    gene track drawn from a GTF file).
    """
    import pandas as pd

    genes, feats = [], []
    for kind, row in iter_gtf(gtf_path, stats=stats, **kwargs):
        (genes if kind == "gene" else feats).append(row)
    return (pd.DataFrame(genes, columns=GENE_COLUMNS),
            pd.DataFrame(feats, columns=FEATURE_COLUMNS))


def import_gtf(
    ds,
    gtf_path,
    *,
    feature_filter=DEFAULT_FEATURES,
    source_label: Optional[str] = None,
    force: bool = False,
    batch_size: int = 10_000,
    verbose: bool = True,
    transcript_id_pattern: Optional[str] = None,
    transcript_id_prefixes=None,
    transcript_tags=None,
    gene_biotypes=None,
    chrom_names="auto",
) -> dict:
    """Parse a GTF into the cytome's annotation tables.

    Reads ``gtf_path`` (transparent ``.gz`` support) with :func:`iter_gtf`, which accepts GENCODE,
    Ensembl, NCBI RefSeq and plain GTF 2.2 alike (module docstring), and writes
    ``_gene_annotation`` (one row per gene, from its ``gene`` line or spanning its transcripts) and
    ``_exon_annotation`` (the other features, with a ``feature`` column). Query them with
    :meth:`CytomeDataset.query_gene_annotation` / :meth:`CytomeDataset.query_exon_annotation`.

    Parameters
    ----------
    ds : CytomeDataset
        Open cytome dataset (write mode).
    gtf_path : str or Path
        Path to GTF file. ``.gz`` extension is detected and
        transparently decompressed.
    feature_filter : tuple[str], default ``('gene', 'exon',
        'transcript', 'CDS', 'UTR', 'five_prime_utr',
        'three_prime_utr')``
        GTF feature types to import. ``'gene'`` makes the gene table (every gene, whether or not
        the file has a ``gene`` line for it); the other types go to ``_exon_annotation`` with a
        ``feature`` column. Round 8 (2026-05-31): expanded from `('gene', 'exon')` to the full set
        so users get correct TSS marks + CDS-priority rendering with zero configuration.
    source_label : str, optional
        Tag written to ``_gene_annotation.source`` for provenance
        (e.g. ``'gencode_vM25'``). Defaults to the GTF's filename
        stem (without ``.gtf`` / ``.gtf.gz`` suffixes).
    force : bool
        If True, wipe existing annotation rows before importing.
        If False (default) and annotation rows already exist,
        raises ``RuntimeError`` with a hint to pass ``force=True``.
    batch_size : int
        Rows per ``executemany`` batch. Larger = faster but more
        RAM. Default 10K.
    verbose : bool
        Print progress and summary counts.
    transcript_id_pattern : str, optional
        Regex applied to each row's ``transcript_id``; rows whose ``transcript_id`` does NOT
        match are skipped (``r'^NM_|^NR_'`` keeps RefSeq curated transcripts). Genes are kept.
    transcript_id_prefixes : list[str], optional
        Shortcut for the prefix-list case: ``['NM_', 'NR_']`` is ``r'^(NM_|NR_)'``. Mutually
        exclusive with ``transcript_id_pattern``.
    transcript_tags : list[str], optional
        Keep only rows of transcripts whose GTF ``tag "X";`` (on any of the transcript's rows)
        matches one of the listed values: ``['basic']`` (GENCODE basic), ``['MANE_Select']``,
        ``['Ensembl_canonical']``. Genes are kept.
    gene_biotypes : list[str], optional
        Keep only genes whose ``gene_type`` / ``gene_biotype`` (from any of the gene's rows) is
        listed, with all their rows: ``['protein_coding']``. Genes whose rows carry no biotype at
        all (NCBI rows other than ``gene`` lines carry none) are dropped and counted in the
        returned ``genes_dropped_no_biotype``.
    chrom_names : ``'auto'``, ``'keep'`` or dict, default ``'auto'``
        ``'auto'``: write chromosome names in the store's own spelling, judged from its
        fragments, peaks and tiles (an Ensembl ``8`` becomes ``chr8`` when the store has
        ``chr8``); names already in the store's spelling, and every name of a store without
        genomic data, are kept. ``'keep'``: as in the file. A dict: an explicit mapping.

    Returns
    -------
    dict
        ``{'n_genes', 'n_exons', 'source'}`` as before (``n_exons`` counts ``exon`` rows only)
        and the reader's counts: ``n_features`` (all non-gene rows), ``features`` (rows per
        type), ``n_gene_lines``, ``n_genes_inferred``, ``n_genes_named``, ``chrom_renamed``, and
        what each filter dropped.
    """
    gtf_path = Path(gtf_path)
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")
    ds.require_writable("cytome.import_gtf")
    conn = ds._conn
    existing_gene_count = conn.execute("SELECT COUNT(*) FROM _gene_annotation").fetchone()[0]
    if existing_gene_count > 0 and not force:
        raise RuntimeError(
            f"cytome.import_gtf: annotation already imported "
            f"({existing_gene_count} genes). Pass force=True to "
            f"replace, or query the existing annotation via "
            f"ds.query_gene_annotation()."
        )
    _transcript_id_regex(transcript_id_pattern, transcript_id_prefixes)   # fail before reading

    if chrom_names == "auto":
        target = store_chromosomes(ds)
        chrom_map = target or None
    elif chrom_names == "keep" or chrom_names is None:
        chrom_map = None
    elif isinstance(chrom_names, dict):
        chrom_map = chrom_names
    else:
        raise ValueError(f"cytome.import_gtf: chrom_names must be 'auto', 'keep' or a dict, got {chrom_names!r}")

    if force:
        conn.execute("DELETE FROM _gene_annotation")
        conn.execute("DELETE FROM _exon_annotation")
        conn.commit()
        if verbose:
            print("cytome.import_gtf: cleared existing annotation tables")
    if verbose:
        print(
            f"cytome.import_gtf: reading {gtf_path.name} "
            f"({gtf_path.stat().st_size / 1e6:.1f} MB), features={sorted(set(feature_filter))}"
        )

    st: dict = {}
    gene_rows, other_rows = [], []

    def _flush():
        if gene_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO _gene_annotation"
                " (gene_id, chrom, start, end, strand, gene_name, gene_type, source)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)", gene_rows)
            gene_rows.clear()
        if other_rows:
            conn.executemany(
                "INSERT INTO _exon_annotation"
                " (gene_id, transcript_id, exon_number, chrom, start, end, strand, feature,"
                "  transcript_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", other_rows)
            other_rows.clear()
        conn.commit()

    for kind, row in iter_gtf(
            gtf_path, feature_filter=feature_filter, source_label=source_label,
            transcript_id_pattern=transcript_id_pattern,
            transcript_id_prefixes=transcript_id_prefixes, transcript_tags=transcript_tags,
            gene_biotypes=gene_biotypes, chrom_map=chrom_map, stats=st):
        (gene_rows if kind == "gene" else other_rows).append(row)
        if len(gene_rows) >= batch_size or len(other_rows) >= batch_size:
            _flush()
    _flush()

    feats = st["features"]
    out = {
        "n_genes": st["genes"],
        "n_exons": int(feats.get("exon", 0)),
        "source": st["source"],
        "n_features": int(sum(feats.values())),
        "features": feats,
        "n_gene_lines": st["gene_lines"],
        "n_genes_inferred": st["genes_inferred"],
        "n_genes_named": st["genes_named"],
        "chrom_renamed": st["chrom_renamed"],
        "genes_split_across_loci": st["genes_split_across_loci"],
        "rows_dropped_transcript_id": st["rows_dropped_transcript_id"],
        "rows_dropped_tag": st["rows_dropped_tag"],
        "rows_dropped_biotype": st["rows_dropped_biotype"],
        "genes_dropped_biotype": st["genes_dropped_biotype"],
        "genes_dropped_no_biotype": st["genes_dropped_no_biotype"],
    }
    ds.metadata["gene_annotation"] = {
        "reader_version": READER_VERSION, "gtf": gtf_path.name,
        "gtf_path": str(gtf_path.resolve()),
        **{k: v for k, v in out.items() if k != "features"}, "features": dict(feats),
        "filters": {"feature_filter": sorted(set(feature_filter)),
                    "transcript_id_pattern": transcript_id_pattern,
                    "transcript_id_prefixes": list(transcript_id_prefixes or []) or None,
                    "transcript_tags": list(transcript_tags or []) or None,
                    "gene_biotypes": list(gene_biotypes or []) or None},
    }
    ds.flush()
    if verbose:
        print(
            f"cytome.import_gtf: done — {out['n_genes']:,} genes ({out['n_gene_lines']:,} gene "
            f"lines, {out['n_genes_inferred']:,} spanned from their transcripts; "
            f"{out['n_genes_named']:,} named), {out['n_features']:,} other features {feats}"
        )
        if out["chrom_renamed"]:
            ex = ", ".join(f"{a}->{b}" for a, b in list(out["chrom_renamed"].items())[:4])
            print(f"  chromosome names in the store's spelling: {len(out['chrom_renamed'])} renamed ({ex}, ...)")
        if out["genes_split_across_loci"]:
            print(f"  {out['genes_split_across_loci']} genes without a gene line have rows on more than "
                  f"one chromosome/strand; each kept on the locus with most rows (e.g. {st['split_examples']})")
        for key, what in [("rows_dropped_transcript_id", "transcript_id filter dropped rows"),
                          ("rows_dropped_tag", "transcript_tags filter dropped rows"),
                          ("genes_dropped_biotype", "gene_biotypes filter dropped genes")]:
            if out[key]:
                print(f"  {what}: {out[key]:,}")
        if out["genes_dropped_no_biotype"]:
            print(f"  of which {out['genes_dropped_no_biotype']:,} carry no biotype on any row")
    from cytome.core.provenance import record_edit
    record_edit(ds, "cytome.import_gtf",
                {"gtf_path": gtf_path, "force": force, "source_label": source_label,
                 "transcript_id_pattern": transcript_id_pattern, "transcript_tags": transcript_tags,
                 "gene_biotypes": gene_biotypes, "chrom_names": chrom_names},
                outputs=["_gene_annotation", "_exon_annotation"])
    return out


# ---------------------------------------------------------------------------
# Biotype backfill & gene table synchronization (cytome 0.4.0)
# ---------------------------------------------------------------------------

def backfill_biotypes_from_exons(ds) -> dict:
    """Infer missing ``_gene_annotation.gene_type`` values from ``_exon_annotation`` in-place.

    RefSeq GTFs without ``gene`` lines on chr2..chrY (such as Cell Ranger ARC Mmul_10 and
    mCalJac1) have ``CDS`` rows in ``_exon_annotation`` for protein-coding genes. This function
    populates ``_gene_annotation.gene_type`` in < 0.2 s without re-reading the external GTF.
    """
    ds.require_writable("cytome.backfill_biotypes_from_exons")
    conn = ds._conn
    n_null_before = conn.execute(
        "SELECT COUNT(*) FROM _gene_annotation WHERE gene_type IS NULL OR gene_type = ''"
    ).fetchone()[0]
    if n_null_before == 0:
        return {"n_null_before": 0, "n_updated_protein_coding": 0, "n_updated_other": 0, "n_null_after": 0}

    cur_pc = conn.execute(
        """
        UPDATE _gene_annotation
        SET gene_type = 'protein_coding'
        WHERE (gene_type IS NULL OR gene_type = '')
          AND gene_id IN (
              SELECT DISTINCT gene_id FROM _exon_annotation
              WHERE feature = 'CDS' OR transcript_type IN ('mRNA', 'protein_coding')
          )
        """
    )
    n_pc = cur_pc.rowcount if cur_pc.rowcount is not None and cur_pc.rowcount >= 0 else 0

    cur_lnc = conn.execute(
        """
        UPDATE _gene_annotation
        SET gene_type = 'lncRNA'
        WHERE (gene_type IS NULL OR gene_type = '')
          AND gene_id IN (
              SELECT DISTINCT gene_id FROM _exon_annotation
              WHERE transcript_type IN ('lnc_RNA', 'lncRNA', 'ncRNA')
          )
        """
    )
    n_other = cur_lnc.rowcount if cur_lnc.rowcount is not None and cur_lnc.rowcount >= 0 else 0

    # Any remaining gene with non-null transcript_type in _exon_annotation
    cur_rem = conn.execute(
        """
        UPDATE _gene_annotation
        SET gene_type = (
            SELECT e.transcript_type FROM _exon_annotation e
            WHERE e.gene_id = _gene_annotation.gene_id
              AND e.transcript_type IS NOT NULL AND e.transcript_type != ''
            LIMIT 1
        )
        WHERE (gene_type IS NULL OR gene_type = '')
          AND EXISTS (
              SELECT 1 FROM _exon_annotation e
              WHERE e.gene_id = _gene_annotation.gene_id
                AND e.transcript_type IS NOT NULL AND e.transcript_type != ''
          )
        """
    )
    if cur_rem.rowcount is not None and cur_rem.rowcount > 0:
        n_other += cur_rem.rowcount

    # Remaining transcripts without CDS on Gnomon/BestRefSeq/RefSeq/Curated Genomic in NCBI RefSeq GTFs
    # are non-coding RNAs (lncRNA) or tRNAs (tRNAscan-SE).
    conn.execute(
        """
        UPDATE _gene_annotation
        SET gene_type = 'tRNA'
        WHERE (gene_type IS NULL OR gene_type = '')
          AND (gene_id LIKE 'TRNA%' OR gene_name LIKE 'TRNA%')
        """
    )
    cur_nc_fallback = conn.execute(
        """
        UPDATE _gene_annotation
        SET gene_type = 'lncRNA'
        WHERE (gene_type IS NULL OR gene_type = '')
          AND EXISTS (
              SELECT 1 FROM _exon_annotation e
              WHERE e.gene_id = _gene_annotation.gene_id
                AND e.feature IN ('transcript', 'exon')
          )
        """
    )
    if cur_nc_fallback.rowcount is not None and cur_nc_fallback.rowcount > 0:
        n_other += cur_nc_fallback.rowcount

    conn.commit()
    n_null_after = conn.execute(
        "SELECT COUNT(*) FROM _gene_annotation WHERE gene_type IS NULL OR gene_type = ''"
    ).fetchone()[0]
    try:
        info = dict(ds.metadata.get("gene_annotation") or {})
        if info:
            info["reader_version"] = max(int(info.get("reader_version", 0) or 0), READER_VERSION)
            ds.metadata["gene_annotation"] = info
            ds.flush()
    except Exception:
        pass
    if n_pc or n_other:
        from cytome.core.provenance import record_edit
        record_edit(ds, "cytome.backfill_biotypes_from_exons", {}, outputs=["_gene_annotation"],
                    inputs=["_exon_annotation"], notes=f"{int(n_pc) + int(n_other)} missing gene_type filled")
    return {
        "n_null_before": int(n_null_before),
        "n_updated_protein_coding": int(n_pc),
        "n_updated_other": int(n_other),
        "n_null_after": int(n_null_after),
    }


def ensure_gtf_imported(
    ds,
    gtf_path=None,
    *,
    copy_from_dataset=None,
    verbose: bool = False,
) -> dict:
    """Ensure ``_gene_annotation`` and ``_exon_annotation`` are populated and have biotypes.

    The reference is data, never searched for:

    1. If ``_gene_annotation`` already has rows, runs :func:`backfill_biotypes_from_exons` if any
       ``gene_type`` is NULL. A ``gtf_path`` given beside it is not read, and a warning says so unless it is
       the file the store's annotation was read from.
    2. Otherwise, if ``gtf_path`` is given, calls :func:`import_gtf` (FileNotFoundError if it does not exist).
    3. Otherwise, if ``copy_from_dataset`` is provided (another open :class:`CytomeDataset` or path
       with the same genome that already has ``_gene_annotation`` populated), copies its
       ``_gene_annotation`` and ``_exon_annotation`` tables.
    4. Otherwise returns ``{'status': 'empty'}``.
    """
    conn = ds._conn
    n_genes = conn.execute("SELECT COUNT(*) FROM _gene_annotation").fetchone()[0]
    if n_genes > 0:
        if gtf_path is not None:
            # The store's own annotation is used, and a GTF given beside it is not read. Said when the two
            # are not the same file, so that a call with another GTF is not taken to have used it.
            info = ds.metadata.get("gene_annotation") or {}
            held = info.get("gtf_path") or info.get("gtf")
            given = Path(gtf_path)
            same = held is not None and (
                str(given.resolve()) == str(held) if info.get("gtf_path") else given.name == str(held))
            if not same:
                import warnings
                warnings.warn(
                    f"the store already holds a gene annotation ({n_genes:,} genes"
                    + (f", read from {held}" if held else "") + f"); gtf_path={str(gtf_path)!r} was not read. "
                    "cytome.import_gtf(ds, gtf_path, force=True) replaces the store's annotation.",
                    UserWarning, stacklevel=3)
        n_null = conn.execute(
            "SELECT COUNT(*) FROM _gene_annotation WHERE gene_type IS NULL OR gene_type = ''"
        ).fetchone()[0]
        bf = {"n_null_before": 0, "n_updated_protein_coding": 0, "n_updated_other": 0, "n_null_after": 0}
        if n_null > 0 and not ds.read_only:
            bf = backfill_biotypes_from_exons(ds)
        return {"status": "existing", "n_genes": int(n_genes), **bf}

    if gtf_path is not None:
        resolved = Path(gtf_path)
        if not resolved.exists():
            raise FileNotFoundError(f"GTF file not found: {resolved}")
        out = import_gtf(ds, resolved, verbose=verbose)
        bf = backfill_biotypes_from_exons(ds)
        return {"status": "imported", **out, **bf}

    if copy_from_dataset is not None:
        ds.require_writable("cytome.ensure_gtf_imported")
        import cytome as _cyt
        close_src = False
        if isinstance(copy_from_dataset, (str, Path)):
            src = _cyt.open(str(copy_from_dataset), mode="r")
            close_src = True
        else:
            src = copy_from_dataset
        try:
            g_rows = src._conn.execute(
                "SELECT gene_id, chrom, start, end, strand, gene_name, gene_type, source FROM _gene_annotation"
            ).fetchall()
            e_rows = src._conn.execute(
                "SELECT gene_id, transcript_id, exon_number, chrom, start, end, strand, feature, transcript_type "
                "FROM _exon_annotation"
            ).fetchall()
            if g_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO _gene_annotation "
                    "(gene_id, chrom, start, end, strand, gene_name, gene_type, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    g_rows,
                )
            if e_rows:
                conn.executemany(
                    "INSERT INTO _exon_annotation "
                    "(gene_id, transcript_id, exon_number, chrom, start, end, strand, feature, transcript_type) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    e_rows,
                )
            conn.commit()
            info = src.metadata.get("gene_annotation")
            if info:
                ds.metadata["gene_annotation"] = dict(info)
                ds.flush()
            bf = backfill_biotypes_from_exons(ds)
            return {"status": "copied", "n_genes": len(g_rows), "n_features": len(e_rows), **bf}
        finally:
            if close_src:
                src.close()

    return {"status": "empty", "n_genes": 0}


_REFSEQ_DUP_RE = re.compile(r"^(.+)_([1-9]\d*)$")
_ENS_VER_RE = re.compile(r"^(ENS[A-Z]*G\d+)(?:\.\d+(?:_PAR_Y)?)?$")


def sync_gene_metadata(
    ds,
    gtf_path=None,
    *,
    copy_from_dataset=None,
    verbose: bool = False,
) -> dict:
    """Synchronize ``genes`` and ``GA_genes`` tables with ``_gene_annotation``.

    Populates NULL ``chr, start, end_, strand, biotype, symbol`` in both ``genes`` and ``GA_genes``
    from ``_gene_annotation``, matching by:
    1. Exact ``gene_id == _gene_annotation.gene_id`` or ``gene_ids == _gene_annotation.gene_id``
    2. Version-stripped Ensembl ID (``ENSG00000123456.12 -> ENSG00000123456``) on ``gene_id`` or
       ``gene_ids`` (handles GENCODE vM23 vs. vM25 symbol renames via stable Ensembl IDs), or RefSeq
       duplicate suffix (``CHRNA2_1 -> CHRNA2``)
    3. Exact ``symbol == _gene_annotation.gene_name`` (or ``gene_id == _gene_annotation.gene_name``),
       choosing the longest canonical chromosome annotation when a symbol appears multiple times.
    """
    ds.require_writable("cytome.sync_gene_metadata")
    gtf_status = ensure_gtf_imported(
        ds, gtf_path=gtf_path, copy_from_dataset=copy_from_dataset, verbose=verbose
    )
    conn = ds._conn
    ann_rows = conn.execute(
        "SELECT gene_id, chrom, start, end, strand, gene_name, gene_type FROM _gene_annotation"
    ).fetchall()
    if not ann_rows:
        return {"gtf": gtf_status, "genes_updated": 0, "GA_genes_updated": 0}

    by_id = {}
    by_stripped_id = {}
    by_name = {}
    for gid, chrom, start, end, strand, gname, gtype in ann_rows:
        rec = (chrom, int(start), int(end), strand or "+", gname or gid, gtype)
        by_id[gid] = rec
        m_ens = _ENS_VER_RE.match(str(gid))
        if m_ens:
            stripped = m_ens.group(1)
            if stripped not in by_stripped_id or not str(gid).endswith("_PAR_Y"):
                by_stripped_id[stripped] = rec
        for key_name in filter(None, (gname, gid)):
            prev = by_name.get(key_name)
            if prev is None:
                by_name[key_name] = rec
            else:
                # Prefer primary chromosomes over scaffolds, then longest span
                prev_primary = not ("_" in str(prev[0]) or str(prev[0]).startswith(("chrUn", "NW_", "NT_")))
                curr_primary = not ("_" in str(chrom) or str(chrom).startswith(("chrUn", "NW_", "NT_")))
                if (curr_primary and not prev_primary) or (
                    curr_primary == prev_primary and (int(end) - int(start)) > (prev[2] - prev[1])
                ):
                    by_name[key_name] = rec

    summary = {"gtf": gtf_status, "genes_updated": 0, "GA_genes_updated": 0}
    for table in ("genes", "GA_genes"):
        cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
        if not cols:
            continue
        if "strand" not in cols:
            conn.execute(f'ALTER TABLE "{table}" ADD COLUMN strand TEXT')
            conn.execute(
                "INSERT OR REPLACE INTO _column_meta(table_name, column_name, dtype, categories) "
                "VALUES (?, 'strand', 'object', NULL)",
                (table,),
            )
        gene_ids_expr = "gene_ids" if "gene_ids" in cols else "NULL AS gene_ids"
        rows = conn.execute(
            f'SELECT gene_idx, gene_id, symbol, chr, start, end_, biotype, strand, {gene_ids_expr} FROM "{table}"'
        ).fetchall()
        if not rows:
            continue
        updates = []
        for gidx, gid, sym, c_chr, c_start, c_end, c_bio, c_strand, g_ens_id in rows:
            rec = by_id.get(gid)
            if rec is None and gid:
                m_ens = _ENS_VER_RE.match(str(gid))
                if m_ens:
                    rec = by_stripped_id.get(m_ens.group(1))
            if rec is None and g_ens_id:
                rec = by_id.get(str(g_ens_id))
                if rec is None:
                    m_ens = _ENS_VER_RE.match(str(g_ens_id))
                    if m_ens:
                        rec = by_stripped_id.get(m_ens.group(1))
                    else:
                        rec = by_stripped_id.get(str(g_ens_id).split(".")[0])
            if rec is None and sym:
                rec = by_name.get(str(sym))
            if rec is None and gid:
                rec = by_name.get(str(gid))
                if rec is None:
                    m_dup = _REFSEQ_DUP_RE.match(str(gid))
                    if m_dup:
                        rec = by_name.get(m_dup.group(1))
            if rec is None:
                continue
            a_chr, a_start, a_end, a_strand, a_name, a_bio = rec
            new_sym = sym if (sym is not None and str(sym) != "") else a_name
            new_chr = c_chr if (c_chr is not None and str(c_chr) != "") else a_chr
            new_start = c_start if c_start is not None else a_start
            new_end = c_end if c_end is not None else a_end
            new_bio = c_bio if (c_bio is not None and str(c_bio) != "") else a_bio
            new_strand = c_strand if (c_strand is not None and str(c_strand) != "") else a_strand
            if (
                new_sym != sym
                or new_chr != c_chr
                or new_start != c_start
                or new_end != c_end
                or new_bio != c_bio
                or new_strand != c_strand
            ):
                updates.append((new_sym, new_chr, new_start, new_end, new_bio, new_strand, gidx))
        if updates:
            conn.executemany(
                f'UPDATE "{table}" SET symbol = ?, chr = ?, start = ?, end_ = ?, biotype = ?, strand = ? '
                f'WHERE gene_idx = ?',
                updates,
            )
            conn.commit()
        summary[f"{table}_updated"] = len(updates)
    changed = [t for t in ("genes", "GA_genes") if summary.get(f"{t}_updated")]
    if changed:
        from cytome.core.provenance import record_edit
        record_edit(ds, "cytome.sync_gene_metadata", {"gtf_path": gtf_path}, outputs=changed,
                    inputs=["_gene_annotation"],
                    notes="missing values filled: " + ", ".join(f"{t} {summary[t + '_updated']}" for t in changed))
    return summary
