"""Optional lazy computed layer descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LazyLayer:
    """Descriptor of a computed-on-read layer.

    This is a lightweight placeholder for future expansion.
    """

    name: str
    base_layer: str
    transform_type: str
    parameters: dict[str, Any]

    def describe(self) -> str:
        """Return human-readable description."""
        return f"{self.name}: {self.transform_type}({self.base_layer})"
