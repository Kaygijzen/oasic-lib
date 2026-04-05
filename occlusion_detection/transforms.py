"""
Transforms for occlusion detection and localization.
"""

import numpy as np
import torch
import cv2
from skimage.filters import threshold_otsu


class LocalizeOcclusion:
    """Custom transform for Localizing any occlusion."""
    # FIXME: hacky wacky
    def __init__(
        self,
        feature_extractor,
        memory_bank,
        grid_size=(16,16),
        device="cuda"
    ):
        self.feature_extractor = feature_extractor
        self.memory_bank = memory_bank
        self.grid_size = grid_size
        self.device = device
        
    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_tensor = image_tensor.to(self.device)
            tokens = self.feature_extractor.get_intermediate_layers(image_tensor)[0]
            if tokens.dim() == 2:  # (N_patches, D) when B=1
                tokens = tokens.unsqueeze(0)
        return tokens.cpu().numpy()  # shape: [B, N_patches, D]

    def score_embedding(self, features):
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        distances, _ = self.memory_bank.kneighbors(features, n_neighbors=1)
        return distances
    
    def compute_anomaly_map(self, img):
        B, C, H, W = img.shape
        features = self.extract_features(img)

        B_feat, T, D = features.shape
        # Reshape to [B*T, D]
        features = features.reshape(-1, D)

        distances = self.score_embedding(features) # [B*T, 1]
        distances = distances.squeeze(1)  # [B * T]

        # Reshape to (B, grid_H, grid_W)
        patch_maps = distances.reshape(B, *self.grid_size)

        # Resize each to full image size (e.g., 224x224)
        pixel_maps = [cv2.resize(patch_map, (W, H), interpolation=cv2.INTER_LINEAR)
                    for patch_map in patch_maps]  # list of (H, W) arrays
        pixel_maps = np.stack(pixel_maps)
        
        # dumb workaround
        pixel_maps = pixel_maps[0] if B == 1 else pixel_maps
        pixel_maps = torch.from_numpy(pixel_maps)

        return pixel_maps
        
    def __call__(self, x):
        
        # TODO: only allow batched?
        # Unpack if input is a tuple
        if isinstance(x, tuple):
            img, *rest = x
        else:
            img, rest = x, []

        if img.dim() < 4:
            # When not yet batched, unsqueeze to [1, C, H, W]
            batched_img = img.unsqueeze(0)
        else:
            batched_img = img
        
        anomaly_map = self.compute_anomaly_map(batched_img)

        # TODO: return batched_image?
        # Repackage with rest of the tuple (if any)
        if rest:
            return (img, anomaly_map, *rest)
        else:
            return img, anomaly_map


class LocalizeAndMaskOcclusion:
    """Localize and mask occlusions using anomaly detection (LoMa).
    
    This transform:
    1. Extracts patch-level features from an image.
    2. Scores their similarity to a reference memory bank.
    3. Builds an anomaly map from the similarity distances.
    4. Masks the anomalous regions in the image with a specified color.
    """

    def __init__(
        self,
        feature_extractor,
        memory_bank,
        masking_color=(127, 127, 127),
        grid_size=(16, 16),
        use_otsu=False,
        masking_threshold=0.5,
        return_anomaly_map=True,
        return_patch_distances=False,
        device="cpu",
    ):
        self.feature_extractor = feature_extractor
        self.memory_bank = memory_bank
        self.masking_color = masking_color
        self.grid_size = grid_size
        self.use_otsu = use_otsu
        self.masking_threshold = masking_threshold
        self.return_anomaly_map = return_anomaly_map
        self.return_patch_distances = return_patch_distances
        self.device = device

    def extract_features(self, image_tensor):
        """Extract patch-level features from the image using the encoder."""
        with torch.inference_mode():
            image_batch = image_tensor.to(self.device)  # shape: [B, 3, H, W]
            tokens = self.feature_extractor.get_intermediate_layers(image_batch)[0]
            if tokens.dim() == 2:  # (N_patches, D) when B=1
                tokens = tokens.unsqueeze(0)
        return tokens.cpu().numpy()  # shape: [B, N_patches, D]

    def score_embedding(self, features):
        """Compute L2 distance to memory bank for each patch embedding."""
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        distances, _ = self.memory_bank.kneighbors(features)

        # if n_neighbors > 1, need to aggregate to a single per-patch score
        distances = distances.mean(axis=1, keepdims=True)

        return distances

    def compute_anomaly_map(self, img):
        B, C, H, W = img.shape
        features = self.extract_features(img)  # [B, N_patches, D]

        B_feat, T, D = features.shape
        # Reshape to [B*T, D]
        features = features.reshape(-1, D)

        patch_distances = self.score_embedding(features) # [B*T, 1]
        patch_distances = torch.from_numpy(patch_distances)
        patch_distances = patch_distances.squeeze(1)  # [B * T]

        # Reshape to (B, grid_H, grid_W)
        patch_distances = patch_distances.reshape(B, *self.grid_size)

        # Resize each to full image size (e.g., 224x224)
        pixel_maps = [cv2.resize(patch_map.numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
                    for patch_map in patch_distances]  # list of (H, W) arrays
        pixel_maps = torch.from_numpy(np.stack(pixel_maps))  # (B, H, W)

        return pixel_maps, patch_distances

    def mask_image(self, img, anomaly_map):
        """
        Mask high-anomaly regions in a batched image tensor.

        Args:
            img: torch.Tensor of shape [B, 3, H, W], normalized image batch
            anomaly_map: torch.Tensor or np.ndarray of shape [B, H, W], unnormalized anomaly maps

        Returns:
            masked_img: torch.Tensor of shape [B, 3, H, W]
            binary_masks: torch.Tensor of shape [B, 1, H, W], float32
        """
        if isinstance(anomaly_map, np.ndarray):
            anomaly_map = torch.from_numpy(anomaly_map)

        anomaly_map = anomaly_map.to(img.device)

        B, _, H, W = img.shape
        binary_masks = []

        for b in range(B):
            amap = anomaly_map[b].detach().cpu().numpy()

            if self.use_otsu:
                threshold = threshold_otsu(amap)
            else:
                threshold = self.masking_threshold

            mask_np = (amap >= threshold).astype(np.uint8)
            mask = torch.from_numpy(mask_np).to(img.device).unsqueeze(0).float()  # [1, H, W]
            binary_masks.append(mask)

        # Stack into [B, 1, H, W]
        mask = torch.stack(binary_masks, dim=0)

        # Convert masking color to a normalized float tensor on the right device
        if isinstance(self.masking_color, torch.Tensor):
            gray = self.masking_color.to(dtype=img.dtype, device=img.device)
        elif isinstance(self.masking_color, tuple) and max(self.masking_color) > 1:
            gray = torch.tensor(self.masking_color, dtype=img.dtype, device=img.device) / 255.0
        else:
            gray = torch.tensor(self.masking_color, dtype=img.dtype, device=img.device)

        gray = gray.view(1, 3, 1, 1).expand(B, -1, H, W)  # [B, 3, H, W]
        # mask_exp = mask.expand(-1, 3, -1, -1)                 # [B, 3, H, W]
        # Apply mask
        masked_img = img * (1 - mask) + gray * mask

        mask = mask.squeeze(1)
        return masked_img, mask

    def __call__(self, x, return_anomaly_only=False):
        """Apply LoMa transform to an image tensor (shape: [B, 3, H, W])."""
        # Unpack if input is a tuple
        if isinstance(x, tuple):
            img, *rest = x
        else:
            img, rest = x, []
        
        if img.dim() < 4:
            # When not yet batched, unsqueeze to [1, C, H, W]
            batched_img = img.unsqueeze(0)
        else:
            batched_img = img
        
        anomaly_map, patch_distances = self.compute_anomaly_map(batched_img)

        if return_anomaly_only:
            return None, None, anomaly_map

        # [H, W] occ_map is the thresholded anomaly_map
        masked_img, occ_map = self.mask_image(batched_img, anomaly_map)

        out = [masked_img, occ_map]

        # Add anomaly map if requested
        if self.return_anomaly_map:
            out.append(anomaly_map)

        # Add patch distances if requested
        if self.return_patch_distances:
            out.append(patch_distances)

        # Add rest if present
        if rest:
            out.extend(rest)

        return tuple(out)