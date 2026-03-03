"""
Core functions for occlusion map generation and processing.
""" 

import torch
import numpy as np
from skimage.filters import threshold_otsu


def anomaly_map_to_occ_map(
    anomaly_map,
    masking_threshold=0.5,
    use_otsu=False,
    otsu_offset = 0
):
    """
        Converts float anomaly map(s) to binary occlusion map(s).
        Supports [H, W] or [B, H, W] tensors.
    """
    if anomaly_map.ndim == 2:  # single image
        if use_otsu:
            thr = threshold_otsu(anomaly_map.cpu().numpy()) + otsu_offset
        else:
            thr = masking_threshold
        return (anomaly_map >= thr).to(torch.uint8)

    elif anomaly_map.ndim == 3:  # batch
        if use_otsu:
            # per-image Otsu thresholding
            masks = []
            thrs = []
            for i in range(anomaly_map.shape[0]):
                thr = threshold_otsu(anomaly_map[i].cpu().numpy()) + otsu_offset
                thrs.append(thr)
                masks.append((anomaly_map[i] >= thr).to(torch.uint8))
            thr = sum(thrs) / len(thrs) if len(thrs) > 0 else 0 # mean threshold
            masks = torch.stack(masks, dim=0) # [B, H, W]
            return masks, thr 
        else:
            return (anomaly_map >= masking_threshold).to(torch.uint8), thr
    else:
        raise ValueError("Expected [H,W] or [B,H,W] tensor")


def batched_anomaly_map_to_occ_map(
    anomaly_map: torch.Tensor,
    masking_threshold: float = 0.5,
    use_otsu: bool = False
) -> torch.Tensor:
    """
    Converts a batched anomaly map to a binary occlusion map via thresholding.

    Args:
        anomaly_map (Tensor): shape (B, H, W) or (B, 1, H, W)
        masking_threshold (float): threshold value to binarize anomaly map
        use_otsu (bool): apply Otsu thresholding (currently not supported in batch)

    Returns:
        occ_map (Tensor): binary map (B, H, W), dtype=torch.uint8
    """
    if anomaly_map.dim() == 4:
        anomaly_map = anomaly_map.squeeze(1)  # (B, H, W)

    if use_otsu:
        raise NotImplementedError("Batched Otsu thresholding is not supported.")

    # Apply thresholding
    occ_map = (anomaly_map >= masking_threshold).to(torch.uint8)

    return occ_map

