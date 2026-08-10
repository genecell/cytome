"""Cytome command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import cytome
from cytome.io.export_bigwig import export_coverage
from cytome.io.merge import merge as merge_func
from cytome.utils.regions import parse_region


def main() -> None:
    parser = argparse.ArgumentParser(prog="cytome", description="Cytome CLI")
    sub = parser.add_subparsers(dest="command")

    p_convert = sub.add_parser("convert", help="Convert between formats")
    p_convert.add_argument("input")
    p_convert.add_argument("output")
    p_convert.add_argument("--modality", "-m", default="RNA")
    p_convert.add_argument("--arc", action="store_true")
    p_convert.add_argument("--no-fragments", action="store_true")

    p_info = sub.add_parser("info", help="Show dataset info")
    p_info.add_argument("path")
    p_info.add_argument("--json", action="store_true")
    p_info.add_argument("--provenance", action="store_true")

    p_merge = sub.add_parser("merge", help="Merge .cytome files")
    p_merge.add_argument("inputs", nargs="+")
    p_merge.add_argument("-o", "--output", required=True)
    p_merge.add_argument("--batch-key", default="sample_id")
    p_merge.add_argument("--genes", default="intersection", choices=["intersection", "union"])
    p_merge.add_argument("--no-fragments", action="store_true")

    p_subset = sub.add_parser("subset", help="Subset cells")
    p_subset.add_argument("input")
    p_subset.add_argument("-o", "--output", required=True)
    p_subset.add_argument("--query")
    p_subset.add_argument("--barcodes")
    p_subset.add_argument("--sample")

    p_down = sub.add_parser("downsample", help="Downsample cells")
    p_down.add_argument("input")
    p_down.add_argument("-o", "--output", required=True)
    p_down.add_argument("--n-cells", type=int)
    p_down.add_argument("--fraction", type=float)
    p_down.add_argument("--stratify")
    p_down.add_argument("--seed", type=int, default=42)

    p_val = sub.add_parser("validate", help="Validate dataset")
    p_val.add_argument("path")

    p_prov = sub.add_parser("provenance", help="Show provenance")
    p_prov.add_argument("path")
    p_prov.add_argument("--methods-text", action="store_true")

    p_frag = sub.add_parser("export-fragments", help="Export fragments")
    p_frag.add_argument("input")
    p_frag.add_argument("-o", "--output", required=True)
    p_frag.add_argument("--groupby")
    p_frag.add_argument("--region")

    p_cov = sub.add_parser("export-coverage", help="Export coverage")
    p_cov.add_argument("input")
    p_cov.add_argument("-o", "--output", required=True)
    p_cov.add_argument("--groupby", required=True)
    p_cov.add_argument("--format", default="bigwig", choices=["bigwig", "bedgraph"])
    p_cov.add_argument("--normalize", default="cpm", choices=["cpm", "rpkm", "raw"])
    p_cov.add_argument("--bin-size", type=int, default=10)
    p_cov.add_argument("--region")

    p_csc = sub.add_parser("build-csc", help="Build CSC index")
    p_csc.add_argument("input")
    p_csc.add_argument("--matrix")
    p_csc.add_argument("--all", action="store_true")

    p_copy = sub.add_parser("copy", help="Copy dataset")
    p_copy.add_argument("input")
    p_copy.add_argument("output")

    p_init = sub.add_parser("init", help="Create cytome from barcodes file")
    p_init.add_argument("--barcodes", required=True, help="Barcodes file (one per line or TSV)")
    p_init.add_argument("-o", "--output", required=True, help="Output .cytome path")
    p_init.add_argument("--sample-id", default=None, help="Sample identifier")

    # Note: bulk fragment import lives in PIASO now (piaso.pp.importFragments /
    # the cytome-import-fragments binary built from PIASO's Rust crate). cytome
    # keeps fragment storage / query / export; for a small pure-Python import
    # use cytome.io.convert_fragments.import_fragments.

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        _dispatch(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "convert":
        _cmd_convert(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "merge":
        _cmd_merge(args)
    elif args.command == "subset":
        _cmd_subset(args)
    elif args.command == "downsample":
        _cmd_downsample(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "provenance":
        _cmd_provenance(args)
    elif args.command == "export-fragments":
        _cmd_export_fragments(args)
    elif args.command == "export-coverage":
        _cmd_export_coverage(args)
    elif args.command == "build-csc":
        _cmd_build_csc(args)
    elif args.command == "copy":
        _cmd_copy(args)
    elif args.command == "init":
        _cmd_init(args)


def _cmd_convert(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)
    if inp.is_dir():
        if args.arc:
            ds = cytome.from_cellranger_arc(
                inp,
                out,
                import_fragments=not args.no_fragments,
            )
            ds.close()
            print("Done.")
            return
        ds = cytome.from_cellranger(inp, out)
        ds.close()
        print("Done.")
        return

    if inp.suffix == ".h5ad" and out.suffix == ".cytome":
        import anndata

        adata = anndata.read_h5ad(str(inp))
        ds = cytome.from_anndata(adata, modality=args.modality, output=str(out))
        ds.close()
    elif inp.suffix == ".cytome" and out.suffix == ".h5ad":
        ds = cytome.open(inp)
        adata = ds.to_anndata(modality=args.modality)
        adata.write_h5ad(str(out))
        ds.close()
    elif inp.suffix == ".h5mu" and out.suffix == ".cytome":
        import mudata

        mdata = mudata.read_h5mu(str(inp))
        ds = cytome.from_mudata(mdata, output=str(out))
        ds.close()
    elif inp.suffix == ".cytome" and out.suffix == ".h5mu":
        ds = cytome.open(inp)
        mdata = ds.to_mudata()
        mdata.write_h5mu(str(out))
        ds.close()
    else:
        raise ValueError(f"Unsupported conversion: {inp} -> {out}")
    print("Done.")


def _cmd_info(args: argparse.Namespace) -> None:
    ds = cytome.open(args.path)
    if args.json:
        print(json.dumps(ds.to_info_dict(), indent=2, default=str))
    elif args.provenance:
        print(ds.provenance.show())
    else:
        print(ds)
    ds.close()


def _cmd_merge(args: argparse.Namespace) -> None:
    ds = merge_func(
        args.inputs,
        output=args.output,
        batch_key=args.batch_key,
        gene_strategy=args.genes,
        include_fragments=not args.no_fragments,
    )
    print(f"Merged dataset: {ds.n_cells} cells")
    ds.close()


def _cmd_subset(args: argparse.Namespace) -> None:
    ds = cytome.open(args.input)
    if args.query:
        selected = ds.cells.query(args.query)
        keep = selected["cell_idx"].to_numpy(dtype=np.int64)
    elif args.barcodes:
        with open(args.barcodes, "rt") as f:
            wanted = {line.strip() for line in f if line.strip()}
        all_barcodes = ds.cells["barcode"]
        keep = np.where(np.isin(all_barcodes, list(wanted)))[0]
    elif args.sample:
        wanted = set(args.sample.split(","))
        sample_id = ds.cells["sample_id"]
        keep = np.where(np.isin(sample_id, list(wanted)))[0]
    else:
        raise ValueError("Specify one of --query, --barcodes, or --sample")
    out = ds.subset(keep, output=args.output)
    out.close()
    ds.close()
    print("Done.")


def _cmd_downsample(args: argparse.Namespace) -> None:
    ds = cytome.open(args.input)
    out = ds.downsample(
        n_cells=args.n_cells,
        fraction=args.fraction,
        method="stratified" if args.stratify else "random",
        groupby=args.stratify,
        seed=args.seed,
        output=args.output,
    )
    out.close()
    ds.close()
    print("Done.")


def _cmd_validate(args: argparse.Namespace) -> None:
    ds = cytome.open(args.path)
    rep = ds.validate()
    print("PASSED" if rep.passed else "FAILED")
    if rep.checks_failed:
        print("Failures:")
        for item in rep.checks_failed:
            print(f"- {item}")
    ds.close()


def _cmd_provenance(args: argparse.Namespace) -> None:
    ds = cytome.open(args.path)
    if args.methods_text:
        print(ds.provenance.export_methods_text())
    else:
        print(ds.provenance.show())
    ds.close()


def _cmd_export_fragments(args: argparse.Namespace) -> None:
    ds = cytome.open(args.input)
    if args.groupby:
        ds.ATAC.fragments.export_by_group(args.groupby, args.output)
    else:
        region = parse_region(args.region) if args.region else None
        ds.ATAC.fragments.export(args.output, region=region)
    ds.close()
    print("Done.")


def _cmd_export_coverage(args: argparse.Namespace) -> None:
    ds = cytome.open(args.input)
    region = parse_region(args.region) if args.region else None
    export_coverage(
        dataset=ds,
        groupby=args.groupby,
        output_dir=args.output,
        format=args.format,
        normalize=args.normalize,
        bin_size=args.bin_size,
        region=region,
    )
    ds.close()
    print("Done.")


def _cmd_build_csc(args: argparse.Namespace) -> None:
    ds = cytome.open(args.input)
    rows = ds._conn.execute("SELECT matrix_name FROM matrix_meta ORDER BY matrix_name").fetchall()
    names = [r[0] for r in rows]
    if args.matrix:
        names = [args.matrix]
    elif not args.all:
        raise ValueError("Provide --matrix or --all")

    for name in names:
        mod, layer = name.split("_", 1)
        ds.__getattr__(mod).layer(layer).build_feature_index()
    ds.flush()
    ds.close()
    print("Done.")


def _cmd_copy(args: argparse.Namespace) -> None:
    ds = cytome.open(args.input)
    out = ds.copy(args.output)
    out.close()
    ds.close()
    print("Done.")


def _cmd_init(args: argparse.Namespace) -> None:
    import pandas as pd

    bc_df = pd.read_csv(args.barcodes, sep="\t", header=None, comment="#", dtype=str)
    bc_df.columns = ["barcode"] + [f"col_{i}" for i in range(1, len(bc_df.columns))]
    bc_df = bc_df[["barcode"]].drop_duplicates()
    if args.sample_id:
        bc_df["sample_id"] = args.sample_id

    ds = cytome.create(args.output, force=True)  # CLI keeps overwrite semantics
    ds.set_entity("cells", bc_df)
    ds.flush()
    ds.close()
    print(f"Created {args.output} with {len(bc_df)} cells")


if __name__ == "__main__":
    main()
