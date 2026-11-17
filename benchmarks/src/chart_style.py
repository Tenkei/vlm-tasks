"""Shared palette and checksum-based colors for benchmark charts."""
from __future__ import annotations

from hashlib import sha256

# Palette order is fixed: changing its length or order changes color assignments.
# SHA-256 slots 5 and 3 preserve the existing Qwen3-VL-2B and 235B colors.
COLORS = (
    "#b44271", "#607d8b", "#138575", "#864b9a", "#a17a16",
    "#2563ad", "#a45145", "#66833a", "#c05822", "#8b7660",
)


def model_color(model):
    """Pick a stable palette slot from the model ID, independent of run selection."""
    checksum = int.from_bytes(sha256(model.encode("utf-8")).digest()[:8], "big")
    return COLORS[checksum % len(COLORS)]


def next_color(color):
    """Distinguish a second comparison curve with the next palette entry."""
    return COLORS[(COLORS.index(color) + 1) % len(COLORS)]
