"""Tests for core occlusion functions."""

import pytest

torch = pytest.importorskip("torch")

from occlusion_generation.functions import (
    get_occlusion_mask,
    apply_occlusion,
    apply_occlusion_from_mask,
)


class TestGetOcclusionMask:
    """Tests for the get_occlusion_mask function."""

    @pytest.fixture
    def dimensions(self):
        return 224, 224

    def test_slide_blackout_type(self, dimensions):
        """Test slide_blackout occlusion type."""
        h, w = dimensions
        mask = get_occlusion_mask(h, w, "slide_blackout", 30)
        assert mask.shape == (h, w)

    def test_bars_blackout_type(self, dimensions):
        """Test bars_blackout occlusion type."""
        h, w = dimensions
        mask = get_occlusion_mask(h, w, "bars_blackout", 30)
        assert mask.shape == (h, w)

    def test_grid_dropout_type(self, dimensions):
        """Test grid_dropout occlusion type."""
        h, w = dimensions
        mask = get_occlusion_mask(h, w, "grid_dropout", 30)
        assert mask.shape == (h, w)

    def test_perlin_type(self, dimensions):
        """Test perlin occlusion type."""
        h, w = dimensions
        mask = get_occlusion_mask(h, w, "perlin", 30)
        assert mask.shape == (h, w)

    def test_unknown_type_raises_error(self, dimensions):
        """Test that unknown occlusion type raises ValueError."""
        h, w = dimensions
        with pytest.raises(ValueError, match="Unknown occlusion type"):
            get_occlusion_mask(h, w, "unknown_type", 30)

    def test_zero_percent_returns_empty(self, dimensions):
        """Test that 0% occlusion returns empty mask."""
        h, w = dimensions
        mask = get_occlusion_mask(h, w, "perlin", 0)
        assert mask.sum() == 0


class TestApplyOcclusion:
    """Tests for the apply_occlusion function."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image tensor."""
        return torch.rand(3, 224, 224)

    def test_returns_tensor(self, sample_image):
        """Test that apply_occlusion returns a tensor."""
        result = apply_occlusion(sample_image, 30, "perlin")
        assert isinstance(result, torch.Tensor)

    def test_output_shape_matches_input(self, sample_image):
        """Test that output shape matches input shape."""
        result = apply_occlusion(sample_image, 30, "perlin")
        assert result.shape == sample_image.shape

    def test_zero_percent_returns_original(self, sample_image):
        """Test that 0% occlusion returns the original image."""
        result = apply_occlusion(sample_image, 0, "perlin")
        assert torch.equal(result, sample_image)

    def test_return_mask_option(self, sample_image):
        """Test that return_mask=True returns a tuple."""
        result = apply_occlusion(sample_image, 30, "perlin", return_mask=True)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestApplyOcclusionFromMask:
    """Tests for the apply_occlusion_from_mask function."""

    def test_applies_mask_correctly(self):
        """Test that mask is applied correctly to image."""
        img = torch.ones(3, 10, 10)
        mask = torch.zeros(10, 10)
        mask[5:, :] = 1  # Occlude bottom half

        result = apply_occlusion_from_mask(img, mask, torch.tensor([0, 0, 0]))
        
        # Top half should be unchanged (all ones)
        assert torch.all(result[:, :5, :] == 1)
        # Bottom half should be occluded (all zeros)
        assert torch.all(result[:, 5:, :] == 0)
