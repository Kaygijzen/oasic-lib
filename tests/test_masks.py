"""Tests for mask generation functions."""

import pytest

torch = pytest.importorskip("torch")

from occlusion_generation.masks import (
    slide_blackout_mask,
    bars_blackout_mask,
    grid_dropout_mask,
    perlin_mask,
)


class TestSlideBlackoutMask:
    """Tests for slide blackout mask generation."""

    def test_returns_correct_shape(self):
        """Test that the mask has the correct dimensions."""
        h, w = 224, 224
        mask = slide_blackout_mask(h, w, 50)
        assert mask.shape == (h, w)

    def test_zero_percent_returns_empty_mask(self):
        """Test that 0% occlusion returns an empty mask."""
        mask = slide_blackout_mask(100, 100, 0)
        assert mask.sum() == 0

    def test_hundred_percent_returns_full_mask(self):
        """Test that 100% occlusion returns a full mask."""
        mask = slide_blackout_mask(100, 100, 100)
        assert mask.sum() == 100 * 100


class TestBarsBlackoutMask:
    """Tests for bars blackout mask generation."""

    def test_returns_correct_shape(self):
        """Test that the mask has the correct dimensions."""
        h, w = 256, 256
        mask = bars_blackout_mask(h, w, 30)
        assert mask.shape == (h, w)

    def test_approximate_coverage(self):
        """Test that the mask covers approximately the right percentage."""
        h, w = 200, 200
        percent = 50
        mask = bars_blackout_mask(h, w, percent)
        actual_percent = (mask.sum().item() / (h * w)) * 100
        # Allow some tolerance due to rounding
        assert abs(actual_percent - percent) < 10


class TestGridDropoutMask:
    """Tests for grid dropout mask generation."""

    def test_returns_correct_shape(self):
        """Test that the mask has the correct dimensions."""
        h, w = 128, 128
        mask = grid_dropout_mask(h, w, 25)
        assert mask.shape == (h, w)


class TestPerlinMask:
    """Tests for Perlin noise mask generation."""

    def test_returns_correct_shape(self):
        """Test that the mask has the correct dimensions."""
        h, w = 224, 224
        mask = perlin_mask(h, w, 40)
        assert mask.shape == (h, w)

    def test_is_binary(self):
        """Test that the mask is binary (0 or 1)."""
        mask = perlin_mask(100, 100, 50)
        unique_values = torch.unique(mask)
        assert all(v in [0, 1] for v in unique_values.tolist())

    def test_reproducible_with_seed(self):
        """Test that the same seed produces the same mask."""
        mask1 = perlin_mask(100, 100, 50, seed=42)
        mask2 = perlin_mask(100, 100, 50, seed=42)
        assert torch.equal(mask1, mask2)
