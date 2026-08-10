"""Genome-related utility functions."""

from __future__ import annotations

from typing import Dict, Tuple

CHROM_ORDER: Dict[str, int] = {
    **{f"chr{i}": i for i in range(1, 23)},
    "chrX": 23,
    "chrY": 24,
    "chrM": 25,
}

_REV_CHROM_ORDER = {v: k for k, v in CHROM_ORDER.items()}


_HG38_SIZES: Dict[str, int] = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415,
    "chrM": 16569,
}

_MM10_SIZES: Dict[str, int] = {
    "chr1": 195471971,
    "chr2": 182113224,
    "chr3": 160039680,
    "chr4": 156508116,
    "chr5": 151834684,
    "chr6": 149736546,
    "chr7": 145441459,
    "chr8": 129401213,
    "chr9": 124595110,
    "chr10": 130694993,
    "chr11": 122082543,
    "chr12": 120129022,
    "chr13": 120421639,
    "chr14": 124902244,
    "chr15": 104043685,
    "chr16": 98207768,
    "chr17": 94987271,
    "chr18": 90702639,
    "chr19": 61431566,
    "chrX": 171031299,
    "chrY": 91744698,
    "chrM": 16299,
}


def chrom_to_int(name: str) -> int:
    """Convert chromosome string to integer code."""
    if name not in CHROM_ORDER:
        raise ValueError(f"Unsupported chromosome name: {name}")
    return CHROM_ORDER[name]


def int_to_chrom(i: int) -> str:
    """Convert chromosome integer code to string name."""
    if i not in _REV_CHROM_ORDER:
        raise ValueError(f"Unsupported chromosome index: {i}")
    return _REV_CHROM_ORDER[i]


def get_chrom_sizes(genome: str = "hg38") -> Dict[str, int]:
    """Return chromosome sizes for a reference genome."""
    g = genome.lower()
    if g == "hg38":
        return dict(_HG38_SIZES)
    if g == "mm10":
        return dict(_MM10_SIZES)
    raise ValueError(f"Unsupported genome: {genome}. Expected hg38 or mm10.")


def parse_region(region_str: str) -> Tuple[str, int, int]:
    """Parse region string ``chr:start-end`` with K/M/G and comma support."""
    from cytome.utils.regions import parse_region as _parse_region

    return _parse_region(region_str)
