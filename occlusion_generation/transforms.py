"""
Occlusion transform classes for applying various occlusion types with 
optional sampling and mask return.
"""

import torch
import random
from .functions import (
    get_occlusion_mask, 
    apply_occlusion,
    paste_vegetation_progressively
)


class BaseOcclusionTransform:
    """Base class for occlusion transforms with sampling support."""

    def __init__(self, occlusion_percent=0, occlusion_type="slide_blackout", return_mask=False, sampling_distribution=None):
        self.occlusion_percent = occlusion_percent
        self.occlusion_type = occlusion_type

        self.return_mask = return_mask

        if sampling_distribution is not None and not isinstance(sampling_distribution, torch.distributions.Distribution):
            raise TypeError("sampling_distribution must be a torch.distributions.Distribution.")
        
        self.sampling_distribution = sampling_distribution

    def get_percent(self):
        """Returns occlusion percent, sampled from distribution if provided."""
        if self.sampling_distribution is not None:
            return int(self.sampling_distribution.sample().clamp(0, 1).item() * 100)
        return self.occlusion_percent


class ApplyGrayOcclusion(BaseOcclusionTransform):
    """Applies grayscale occlusion with fixed or sampled occlusion percent."""

    def __init__(self, occlusion_percent=0, occlusion_type="slide_blackout", occlusion_color=(0, 0, 0), return_mask=False, sampling_distribution=None):
        super().__init__(occlusion_percent, occlusion_type, return_mask, sampling_distribution)
        self.occlusion_color = occlusion_color

    def __call__(self, img):
        _, h, w = img.shape
        occlusion_percent = self.get_percent()

        mask = torch.zeros((h, w))
        if occlusion_percent > 0:
            img = apply_occlusion(img, occlusion_percent, self.occlusion_type, self.occlusion_color, return_mask=self.return_mask)

            if self.return_mask:
                img, mask = img
        
        return (img, mask) if self.return_mask else img
    
    
class ApplyFromToGrayOcclusion(BaseOcclusionTransform):
    """Applies grayscale occlusion in range [0, occlusion_percent) in increments of 10."""

    def __init__(self, occlusion_percent=0, occlusion_type="slide_blackout", occlusion_color=(0, 0, 0), return_mask=False, sampling_distribution=None):
        super().__init__(occlusion_percent, occlusion_type, return_mask, sampling_distribution)
        self.occlusion_color = occlusion_color

    def __call__(self, img):
        _, h, w = img.shape
        occlusion_percent = random.randint(0, self.occlusion_percent / 10) * 10

        mask = torch.zeros((h, w))
        if occlusion_percent > 0:
            img = apply_occlusion(img, occlusion_percent, self.occlusion_type, self.occlusion_color, return_mask=self.return_mask)

            if self.return_mask:
                img, mask = img
        
        return (img, mask) if self.return_mask else img


class ApplyOverlayOcclusion(BaseOcclusionTransform):
    """Applies vegetation overlay occlusion with optional mask return."""

    def __init__(self, overlays, occlusion_percent=0, occlusion_type="slide_blackout", return_mask=False, sampling_distribution=None):
        super().__init__(occlusion_percent, occlusion_type, return_mask, sampling_distribution)
        self.overlays = overlays
        self.return_mask = return_mask

    def __call__(self, img):
        _, h, w = img.shape
        occlusion_percent = self.get_percent()

        mask = torch.zeros((h, w))
        if occlusion_percent > 0:
            mask = get_occlusion_mask(h, w, self.occlusion_type, occlusion_percent)
            img, _ = paste_vegetation_progressively(img, mask, self.overlays)

        return (img, mask) if self.return_mask else img
