"""
Functions to generate various types of occlusion masks.
"""

import torch
import random
from noise import pnoise2


def slide_blackout_mask(h, w, occlusion_percent):
    """
    Returns a binary mask (H, W) with occlusion on the left or right side.
    """
    mask = torch.zeros((h, w), dtype=torch.uint8)
    occlusion_width = int(w * occlusion_percent / 100)
    if occlusion_width > 0:
        if random.random() < 0.5:
            # Left side occlusion
            mask[:, :occlusion_width] = 1
        else:
            # Right side occlusion
            mask[:, -occlusion_width:] = 1
    return mask


def bars_blackout_mask(h, w, occlusion_percent):
    """
    Returns a binary mask (H, W) with vertical bars occluded.
    """
    mask = torch.zeros((h, w), dtype=torch.uint8)
    if occlusion_percent >= 100:
        mask[:, :] = 1
        return mask

    total_occlusion_pixels = int(w * occlusion_percent / 100)
    num_bars = max(6, int(occlusion_percent / 10))
    bar_width = max(1, total_occlusion_pixels // num_bars)

    while bar_width * num_bars > total_occlusion_pixels and num_bars > 1:
        num_bars -= 1

    bar_width = total_occlusion_pixels // num_bars
    last_bar_extra = total_occlusion_pixels - (bar_width * (num_bars - 1))
    available_width = w - total_occlusion_pixels
    spacing = available_width // (num_bars + 1)

    for i in range(num_bars - 1):
        start_pos = spacing * (i + 1) + bar_width * i
        end_pos = start_pos + bar_width
        mask[:, start_pos:end_pos] = 1

    start_pos = spacing * num_bars + bar_width * (num_bars - 1)
    end_pos = start_pos + last_bar_extra
    mask[:, start_pos:end_pos] = 1

    return mask


def grid_dropout_mask(h, w, occlusion_percent):
    """
    Returns a binary mask (H, W) with randomly dropped grid cells.
    """
    mask = torch.zeros((h, w), dtype=torch.uint8)
    if occlusion_percent >= 100:
        mask[:, :] = 1
        return mask

    grid_h = max(2, int(h / 10))
    grid_w = max(2, int(w / 10))
    cells_h = (h + grid_h - 1) // grid_h
    cells_w = (w + grid_w - 1) // grid_w
    total_cells = cells_h * cells_w
    cells_to_drop = int(total_cells * occlusion_percent / 100)

    all_cells = [(i, j) for i in range(cells_h) for j in range(cells_w)]
    random.shuffle(all_cells)

    for i, j in all_cells[:cells_to_drop]:
        y_start = i * grid_h
        y_end = min(y_start + grid_h, h)
        x_start = j * grid_w
        x_end = min(x_start + grid_w, w)
        mask[y_start:y_end, x_start:x_end] = 1

    return mask


def perlin_mask(h, w, occlusion_percent, scale=60.0, octaves=2, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Returns a binary mask (H, W) of realistic blobs using Perlin noise.
    """
    if seed is None:
        seed = random.randint(0, 400)
    noise = torch.zeros((h, w))
    for i in range(h):
        for j in range(w):
            noise[i, j] = pnoise2(i / scale,
                                  j / scale,
                                  octaves=octaves,
                                  persistence=persistence,
                                  lacunarity=lacunarity,
                                  repeatx=w,
                                  repeaty=h,
                                  base=seed)
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    threshold = torch.quantile(noise, 1 - (occlusion_percent / 100.0))
    mask = (noise >= threshold).float()  # 1 = occluded, 0 = visible

    return mask
