"""
Utility functions for tensor manipulation, normalization, and cropping.
"""

import torch


def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Unnormalized a normalized tensor.

    Args:
        tensor (Tensor): shape (C, H, W)
        mean (tuple): mean, default is DINOv2 (0.485, 0.465, 0.406)
        std (tuple): std, default is DINOv2 (0.229, 0.224, 0.225)

    Returns:
        tensor (Tensor): unnormalized tensor map (C, H, W)
    """
    # tensor shape: (C, H, W) or (N, C, H, W)
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def normalize_pixel(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Normalized a single pixel tensor (3).

    Args:
        tensor (Tensor): shape (3)
        mean (tuple): mean, default is DINOv2 (0.485, 0.465, 0.406)
        std (tuple): std, default is DINOv2 (0.229, 0.224, 0.225)

    Returns:
        tensor (Tensor): normalized pixel (3)
    """
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    return (tensor - mean) / std  # shape: (3,)


def center_crop_reshape(tensor_batch, size):
    B, H, W = tensor_batch.shape
    new_h, new_w = size
    
    # flatten spatial dims
    flat = tensor_batch.view(B, H * W)
    
    # compute crop indices
    top = (H - new_h) // 2
    left = (W - new_w) // 2
    
    # reshape back and crop
    tensor_batch_reshaped = tensor_batch.view(B, H, W)
    return tensor_batch_reshaped[:, top:top+new_h, left:left+new_w]