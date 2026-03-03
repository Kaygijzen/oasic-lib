"""
Core functions for occlusion map generation and processing.
"""

import torch
import random
from .masks import (
    slide_blackout_mask,
    bars_blackout_mask,
    grid_dropout_mask,
    perlin_mask,
)


def get_occlusion_mask(h, w, occlusion_type="slide_blackout", occlusion_percent=0):
    """
    Generate various occlusion patterns to an image.

    Parameters:
    h (int): Mask height
    w (int): Mask width
    occlusion_percent (float): Percentage of the image to occlude (0-100)
    occlusion_type (str): Type of occlusion pattern to apply:
                          "slide_blackout" - Occludes right side of image
                          "bars_blackout" - Adds vertical black bars
                          "grid_dropout" - Drops out grid cells from the image
                          
    Returns:
    torch.Tensor: Binary mask with the occlusion pattern
    """
    
    mask = torch.zeros((h,w))

    if occlusion_percent <= 0:
        return mask

    if occlusion_percent >= 100:
        # Set entire image to specified color
        mask[:] = 1


    if occlusion_type == "slide_blackout":
        mask = slide_blackout_mask(h, w, occlusion_percent)
    
    elif occlusion_type == "bars_blackout":
        mask = bars_blackout_mask(h, w, occlusion_percent)

    # elif occlusion_type == "random_rain":
    # elif occlusion_type == "random_snow":

    elif occlusion_type == "grid_dropout":
        mask = grid_dropout_mask(h, w, occlusion_percent)

    elif occlusion_type == "perlin":
        mask = perlin_mask(h, w, occlusion_percent)

    else:
        raise ValueError(f"Unknown occlusion type: {occlusion_type}")
    
    return mask


def apply_occlusion_from_mask(
    img, mask, occlusion_color=torch.tensor((0,0,0))
):
    # Create a copy to avoid modifying the original tensor
    img = img.clone()
    assert isinstance(img, torch.Tensor), "Input must be a torch tensor"

    # Get the number of channels in the image
    c = img.shape[0]

    # Handle the special case where occlusion_color is 'mean'
    if occlusion_color == "mean":
        occlusion_values = img.mean(dim=(1, 2)).clone().detach() # [C]
    else:
        # Prepare occlusion color tensor with the correct number of channels
        # Pad or truncate as needed
        if c <= len(occlusion_color):
            # Use only the first c values
            occlusion_values = occlusion_color[:c]
        else:
            # Pad with zeros if more channels than color values
            occlusion_values = torch.tensor(
                list(occlusion_color) + [0] * (c - len(occlusion_color)),
                device=img.device,
                dtype=img.dtype,
            )
 
    # Ensure mask is boolean
    mask = mask.bool()  # [401, 600]

    # Expand mask to 3 channels
    # mask_3ch = mask.unsqueeze(0).expand(3, -1, -1)  # [3, 401, 600]

    # Create a gray tensor of the same shape as the image
    gray_tensor = torch.ones_like(img) * occlusion_values.view(3, 1, 1)  # [3, 401, 600]

    # Apply the mask: where mask is True, use occlusion value
    masked_image = torch.where(mask, gray_tensor, img)

    return masked_image


def apply_occlusion(
    img, occlusion_percent=0, occlusion_type="slide_blackout", occlusion_color=(0, 0, 0), return_mask=False
):
    """
    Apply various occlusion patterns to an image.

    Parameters:
    img (torch.Tensor): Input image tensor in CHW format
    occlusion_percent (float): Percentage of the image to occlude (0-100)
    occlusion_type (str): Type of occlusion pattern to apply:
                          "slide_blackout" - Occludes right side of image
                          "bars_blackout" - Adds vertical black bars
                          "grid_dropout" - Drops out grid cells from the image
                          "perlin" - Uses thresholded Perlin noise to occlude
                          "mix" - Randomly samples one of above occlusion patterns
    occlusion_color (tuple): RGB color to use for occlusion (default: black (0,0,0))

    Returns:
    torch.Tensor: Image with occlusion applied
    """
    assert isinstance(img, torch.Tensor), "Input must be a torch tensor"

    if occlusion_percent <= 0:
        return img

    img = img.clone()
    c, h, w = img.shape

    # Compute occlusion color
    if occlusion_color == "mean":
        occlusion_values = img.mean(dim=(1, 2))
    else:
        occlusion_values = torch.tensor(
            list(occlusion_color)[:c] + [0] * max(0, c - len(occlusion_color)),
            device=img.device, dtype=img.dtype
        )
    occlusion_values = occlusion_values.view(-1, 1, 1)

    if occlusion_percent >= 100:
        # Set entire image to specified color
        img[:] = occlusion_values
        if return_mask:
            mask = torch.ones((h,w))
            return img, mask
        else:
            return img
    
    if occlusion_type == "mix":
        occlusion_type = random.choice([
            "slide_blackout",
            "bars_blackout",
            "grid_dropout",
            "perlin"
        ])

    mask = get_occlusion_mask(h, w, occlusion_type, occlusion_percent)

    img = apply_occlusion_from_mask(
        img, mask, occlusion_color=occlusion_values
    )
    return (img, mask) if return_mask else img


def paste_vegetation_progressively(image, mask, vegetation_cutouts, pixel_threshold=1000, device="cpu"):
    """
    Apply various occlusion patterns to an image.

    Parameters:
    imag (torch.Tensor): Input image to occlude in CWH format
    mask (torch.Tensor): Binary mask (1: region to occlude) in HW format
    vegetation_cutouts (list(torch.Tensor)): List of RGBA vegetation images in CHW format
    threshold (int): Number of pixels that may be non-occluded in the occlusion mask

    Returns:
    torch.Tensor: Occluded image
    torch.Tensor: Leftover binary mask 
    """
    _, H, W = image.shape
    image = image.clone()
    mask = mask.clone()

    while mask.sum() > pixel_threshold:
        # Pick random cutout and convert to tensor
        veg = random.choice(vegetation_cutouts) # [4, h, w]
        rgb = veg[:3]
        alpha = veg[3] > 0 # binary mask [h, w]
        
        h, w = alpha.shape

        # Random top-left position
        top = random.randint(0, H - 1)
        left = random.randint(0, W - 1)

        # Determine valid paste area
        bottom = min(top + h, H)
        right = min(left + w, W)
        paste_h = bottom - top
        paste_w = right - left

        if paste_h <= 0 or paste_w <= 0:
            continue

        # Crop relevant regions
        alpha_crop = alpha[:paste_h, :paste_w]
        rgb_crop = rgb[:, :paste_h, :paste_w]
        mask_crop = mask[top:bottom, left:right]

        alpha_crop = alpha_crop.to(device)
        rgb_crop = rgb_crop.to(device)

        # Create a paste mask: only where both alpha and original mask are 1
        paste_mask = alpha_crop & mask_crop.bool()  # [h, w]

        if paste_mask.sum() == 0:
            continue

        # Apply paste
        for c in range(3):
            image[c, top:bottom, left:right][paste_mask] = rgb_crop[c][paste_mask]

        # Update binary mask
        mask[top:bottom, left:right][paste_mask] = False

    return image, mask

