"""Tests for transform classes."""

import pytest

torch = pytest.importorskip("torch")

from occlusion_generation.transforms import (
    BaseOcclusionTransform,
    ApplyGrayOcclusion,
    ApplyFromToGrayOcclusion,
)


class TestBaseOcclusionTransform:
    """Tests for BaseOcclusionTransform class."""

    def test_default_values(self):
        """Test default initialization values."""
        transform = BaseOcclusionTransform()
        assert transform.occlusion_percent == 0
        assert transform.occlusion_type == "slide_blackout"
        assert transform.return_mask is False

    def test_get_percent_returns_fixed_value(self):
        """Test get_percent returns fixed value when no distribution."""
        transform = BaseOcclusionTransform(occlusion_percent=50)
        assert transform.get_percent() == 50

    def test_invalid_distribution_raises_error(self):
        """Test that invalid distribution type raises TypeError."""
        with pytest.raises(TypeError):
            BaseOcclusionTransform(sampling_distribution="invalid")


class TestApplyGrayOcclusion:
    """Tests for ApplyGrayOcclusion transform."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image tensor."""
        return torch.rand(3, 64, 64)

    def test_returns_tensor(self, sample_image):
        """Test that transform returns a tensor."""
        transform = ApplyGrayOcclusion(occlusion_percent=30, occlusion_type="perlin")
        result = transform(sample_image)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self, sample_image):
        """Test that output shape matches input."""
        transform = ApplyGrayOcclusion(occlusion_percent=30, occlusion_type="perlin")
        result = transform(sample_image)
        assert result.shape == sample_image.shape

    def test_return_mask(self, sample_image):
        """Test that return_mask option works."""
        transform = ApplyGrayOcclusion(
            occlusion_percent=30, 
            occlusion_type="perlin",
            return_mask=True
        )
        result = transform(sample_image)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_zero_percent_unchanged(self, sample_image):
        """Test that 0% occlusion leaves image unchanged."""
        transform = ApplyGrayOcclusion(occlusion_percent=0)
        result = transform(sample_image)
        assert torch.equal(result, sample_image)


class TestApplyFromToGrayOcclusion:
    """Tests for ApplyFromToGrayOcclusion transform."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image tensor."""
        return torch.rand(3, 64, 64)

    def test_returns_tensor(self, sample_image):
        """Test that transform returns a tensor."""
        transform = ApplyFromToGrayOcclusion(occlusion_percent=50, occlusion_type="perlin")
        result = transform(sample_image)
        assert isinstance(result, torch.Tensor)
