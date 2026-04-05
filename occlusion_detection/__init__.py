"""
Occlusion Detection Module

This module provides tools for detecting and localizing occlusions in images
using anomaly detection techniques.
"""

from .functions import (
    anomaly_map_to_occ_map,
    batched_anomaly_map_to_occ_map,
)

from .transforms import (
    LocalizeOcclusion,
    LocalizeAndMaskOcclusion,
)

__all__ = [
    # Functions
    "anomaly_map_to_occ_map",
    "batched_anomaly_map_to_occ_map",
    # Transforms
    "LocalizeOcclusion",
    "LocalizeAndMaskOcclusion",
]
