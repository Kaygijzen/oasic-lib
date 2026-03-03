"""
Occlusion Generation Module

This module provides tools for generating synthetic occlusions on images,
including mask generation and occlusion application transforms.
"""

from .masks import (
    slide_blackout_mask,
    bars_blackout_mask,
    grid_dropout_mask,
    perlin_mask,
)

from .functions import (
    get_occlusion_mask,
    apply_occlusion,
    apply_occlusion_from_mask,
    paste_vegetation_progressively,
)

from .transforms import (
    BaseOcclusionTransform,
    ApplyGrayOcclusion,
    ApplyFromToGrayOcclusion,
    ApplyOverlayOcclusion,
)

__all__ = [
    # Mask generation
    "slide_blackout_mask",
    "bars_blackout_mask",
    "grid_dropout_mask",
    "perlin_mask",
    # Functions
    "get_occlusion_mask",
    "apply_occlusion",
    "apply_occlusion_from_mask",
    "paste_vegetation_progressively",
    # Transforms
    "BaseOcclusionTransform",
    "ApplyGrayOcclusion",
    "ApplyFromToGrayOcclusion",
    "ApplyOverlayOcclusion",
]
