"""Utility helpers for Cytome."""

from cytome.utils.validation import ValidationReport, repair, validate
from cytome.utils.versioning import CURRENT_VERSION, check_compatibility, migrate_if_needed
from cytome.utils.genome import CHROM_ORDER, chrom_to_int, get_chrom_sizes, int_to_chrom
from cytome.utils.regions import extend_region, merge_regions, parse_region, regions_overlap
from cytome.utils.modality import (
    MODALITY_REGISTRY,
    MODALITY_VAR_ENTITY,
    modality_var_entity,
    modality_feature_table_info,
    modality_has_feature,
    read_feature_column,
    modality_cell_depth,
)

__all__ = [
    "ValidationReport",
    "validate",
    "repair",
    "CURRENT_VERSION",
    "check_compatibility",
    "migrate_if_needed",
    "CHROM_ORDER",
    "chrom_to_int",
    "int_to_chrom",
    "get_chrom_sizes",
    "parse_region",
    "regions_overlap",
    "merge_regions",
    "extend_region",
    "MODALITY_REGISTRY",
    "MODALITY_VAR_ENTITY",
    "modality_var_entity",
    "modality_feature_table_info",
    "modality_has_feature",
    "read_feature_column",
    "modality_cell_depth",
]
