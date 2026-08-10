"""GTF → cytome annotation table import (cytome 0.2.2+).

Parses a GTF file (gzipped or not), converts 1-based closed
coordinates to 0-based half-open (BED convention), and writes
rows to the ``_gene_annotation`` and ``_exon_annotation`` tables.

The annotation lives in the cytome and is queryable via
``ds.query_gene_annotation()`` / ``ds.query_exon_annotation()``.
"""
from __future__ import annotations

import gzip
import os
import re
from pathlib import Path
from typing import Optional


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


def import_gtf(
    ds,
    gtf_path,
    *,
    feature_filter=(
        "gene", "exon", "transcript",
        "CDS", "UTR",
        "five_prime_utr", "three_prime_utr",
    ),
    source_label: Optional[str] = None,
    force: bool = False,
    batch_size: int = 10_000,
    verbose: bool = True,
    transcript_id_pattern: Optional[str] = None,
    transcript_id_prefixes=None,
    transcript_tags=None,
    gene_biotypes=None,
) -> dict:
    """Parse a GTF into the cytome's annotation tables.

    Reads ``gtf_path`` (transparent ``.gz`` support), extracts the
    requested feature types, converts 1-based closed coordinates
    to 0-based half-open (BED convention), and writes rows to
    ``_gene_annotation`` (feature='gene') and ``_exon_annotation``
    (other features). The cytome can then be queried via
    :meth:`CytomeDataset.query_gene_annotation` and
    :meth:`CytomeDataset.query_exon_annotation` without re-parsing
    the GTF.

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
        GTF feature types to import. ``'gene'`` rows go to
        ``_gene_annotation``; all other feature types go to
        ``_exon_annotation`` with a ``feature`` column recording
        which type they are.

        Round 8 (2026-05-31): expanded from `('gene', 'exon')` to
        the full set so users get correct TSS marks + CDS-priority
        rendering with zero configuration. Pass a narrower
        filter if you don't need the extras (annotation tables
        get ~2-3× smaller).
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
        Regex (passed to ``re.compile``) applied to each row's
        ``transcript_id`` attribute. Rows whose ``transcript_id``
        does NOT match are skipped. ``gene`` rows (no transcript_id)
        are always kept. Use e.g. ``r'^NM_|^NR_'`` to keep only
        RefSeq curated transcripts. ``None`` (default) = no filter.
    transcript_id_prefixes : list[str], optional
        Friendlier shortcut for the common prefix-list case:
        ``transcript_id_prefixes=['NM_', 'NR_']`` is equivalent to
        ``transcript_id_pattern=r'^(NM_|NR_)'``. Mutually exclusive
        with ``transcript_id_pattern`` — pass one or the other.
    transcript_tags : list[str], optional
        Keep only transcript-row child records whose GTF
        ``tag "X";`` attribute matches one of the listed values.
        Examples: ``['basic']`` (GENCODE basic subset),
        ``['MANE_Select']`` (cross-source canonical), or
        ``['Ensembl_canonical']``. Cascades through child
        ``exon`` / ``UTR`` / ``CDS`` rows of filtered transcripts
        (rows linked by ``transcript_id``).
    gene_biotypes : list[str], optional
        Keep only genes whose ``gene_biotype`` (Ensembl /GENCODE)
        or ``gene_type`` (GENCODE) attribute matches. Examples:
        ``['protein_coding']``, ``['protein_coding', 'lncRNA']``.
        Cascades to child rows whose ``gene_id`` matches the
        filtered gene set.

    Returns
    -------
    dict
        ``{'n_genes': int, 'n_exons': int, 'source': str}``.
    """
    gtf_path = Path(gtf_path)
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")

    if source_label is None:
        # Strip .gtf and/or .gz suffixes
        source_label = gtf_path.name
        for suffix in (".gz", ".gtf"):
            if source_label.endswith(suffix):
                source_label = source_label[: -len(suffix)]

    conn = ds._conn

    # Check for existing annotation
    existing_gene_count = conn.execute(
        "SELECT COUNT(*) FROM _gene_annotation"
    ).fetchone()[0]
    if existing_gene_count > 0 and not force:
        raise RuntimeError(
            f"cytome.import_gtf: annotation already imported "
            f"({existing_gene_count} genes). Pass force=True to "
            f"replace, or query the existing annotation via "
            f"ds.query_gene_annotation()."
        )

    if force:
        conn.execute("DELETE FROM _gene_annotation")
        conn.execute("DELETE FROM _exon_annotation")
        conn.commit()
        if verbose:
            print("cytome.import_gtf: cleared existing annotation tables")

    feature_filter_set = set(feature_filter)

    # Build the transcript_id filter regex. Accept either an
    # explicit regex string or a list of prefixes (friendlier API).
    if transcript_id_pattern is not None and transcript_id_prefixes is not None:
        raise ValueError(
            "cytome.import_gtf: pass either transcript_id_pattern "
            "OR transcript_id_prefixes, not both."
        )
    tx_id_regex = None
    if transcript_id_prefixes:
        tx_id_regex = re.compile(
            "^(" + "|".join(re.escape(p) for p in transcript_id_prefixes) + ")"
        )
    elif transcript_id_pattern is not None:
        tx_id_regex = re.compile(transcript_id_pattern)

    transcript_tags_set = set(transcript_tags) if transcript_tags else None
    gene_biotypes_set = set(gene_biotypes) if gene_biotypes else None

    # Cascading sets — populated on the first sight of a gene /
    # transcript line, then used to filter child rows.
    allowed_gene_ids = (
        None if gene_biotypes_set is None else set()
    )
    allowed_tx_ids = (
        None if transcript_tags_set is None else set()
    )

    gene_rows = []
    other_rows = []
    n_genes = 0
    n_others = 0
    n_skipped_tx_filter = 0
    n_skipped_biotype = 0
    n_skipped_tag = 0

    if verbose:
        size = gtf_path.stat().st_size
        print(
            f"cytome.import_gtf: reading {gtf_path.name} "
            f"({size / 1e6:.1f} MB), features={sorted(feature_filter_set)}, "
            f"source={source_label!r}"
        )

    opener = gzip.open if str(gtf_path).endswith(".gz") else open

    def _flush_batches():
        nonlocal gene_rows, other_rows, n_genes, n_others
        if gene_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO _gene_annotation"
                " (gene_id, chrom, start, end, strand, gene_name,"
                "  gene_type, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                gene_rows,
            )
            n_genes += len(gene_rows)
            gene_rows = []
        if other_rows:
            conn.executemany(
                "INSERT INTO _exon_annotation"
                " (gene_id, transcript_id, exon_number, chrom,"
                "  start, end, strand, feature, transcript_type)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                other_rows,
            )
            n_others += len(other_rows)
            other_rows = []
        conn.commit()

    with opener(gtf_path, "rt") as f:
        for line_no, line in enumerate(f, start=1):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature not in feature_filter_set:
                continue
            chrom = fields[0]
            try:
                gtf_start = int(fields[3])
                gtf_end = int(fields[4])
            except ValueError:
                continue
            # GTF: 1-based closed → cytome: 0-based half-open
            start = gtf_start - 1
            end = gtf_end
            strand = fields[6] if fields[6] in ("+", "-", ".") else "."
            attrs = _parse_attributes(fields[8])
            gene_id = attrs.get("gene_id", "")
            if not gene_id:
                continue
            gene_name = attrs.get("gene_name") or attrs.get("Name")
            gene_type = attrs.get("gene_type") or attrs.get("gene_biotype")

            if feature == "gene":
                # Apply gene_biotypes filter at the gene line. The
                # set of accepted gene_ids cascades to child rows.
                if gene_biotypes_set is not None:
                    if gene_type not in gene_biotypes_set:
                        n_skipped_biotype += 1
                        continue
                    allowed_gene_ids.add(gene_id)
                gene_rows.append((
                    gene_id, chrom, start, end, strand,
                    gene_name, gene_type, source_label,
                ))
            else:
                transcript_id = attrs.get("transcript_id")
                # Cascade gene_biotypes filter to child rows
                if (
                    allowed_gene_ids is not None
                    and gene_id not in allowed_gene_ids
                ):
                    n_skipped_biotype += 1
                    continue
                # Apply transcript_id filter (cascades to exon /
                # UTR / CDS rows of filtered transcripts). Rows
                # without a transcript_id (rare for non-gene
                # features) are kept.
                if (
                    tx_id_regex is not None
                    and transcript_id
                    and not tx_id_regex.match(transcript_id)
                ):
                    n_skipped_tx_filter += 1
                    continue
                # transcript_tags filter — on transcript lines
                # check the tags directly; on exon/UTR/CDS lines
                # cascade via the allowed_tx_ids set.
                if transcript_tags_set is not None:
                    if feature == "transcript":
                        tags = _parse_attribute_tags(fields[8], "tag")
                        if not any(t in transcript_tags_set for t in tags):
                            n_skipped_tag += 1
                            continue
                        if transcript_id:
                            allowed_tx_ids.add(transcript_id)
                    else:
                        # child row — must belong to an allowed transcript
                        if (
                            transcript_id is None
                            or transcript_id not in allowed_tx_ids
                        ):
                            n_skipped_tag += 1
                            continue
                exon_num = attrs.get("exon_number")
                try:
                    exon_num = int(exon_num) if exon_num else None
                except ValueError:
                    exon_num = None
                # transcript_type (GENCODE) / transcript_biotype (Ensembl)
                # — saved per row so users can filter transcripts by
                # biotype at query time (Round 8).
                transcript_type = (
                    attrs.get("transcript_type")
                    or attrs.get("transcript_biotype")
                )
                other_rows.append((
                    gene_id, transcript_id, exon_num,
                    chrom, start, end, strand, feature,
                    transcript_type,
                ))

            if len(gene_rows) >= batch_size or len(other_rows) >= batch_size:
                _flush_batches()
                if verbose and (line_no % 100_000 == 0):
                    print(
                        f"  ... line {line_no:,}: "
                        f"genes={n_genes:,}, other={n_others:,}"
                    )

    _flush_batches()

    if verbose:
        print(
            f"cytome.import_gtf: done — {n_genes:,} genes, "
            f"{n_others:,} other features ({sorted(feature_filter_set - {'gene'})})"
        )
        if tx_id_regex is not None:
            print(
                f"  transcript_id filter dropped "
                f"{n_skipped_tx_filter:,} child rows"
            )
        if gene_biotypes_set is not None:
            print(
                f"  gene_biotypes filter dropped "
                f"{n_skipped_biotype:,} rows"
            )
        if transcript_tags_set is not None:
            print(
                f"  transcript_tags filter dropped "
                f"{n_skipped_tag:,} rows"
            )

    return {
        "n_genes": n_genes,
        "n_exons": n_others,
        "source": source_label,
    }
