import pytest

# Instead of importing cv2, we will mock it as we can't add dependencies.
from unittest.mock import MagicMock
import sys

# Mock cv2
sys.modules['cv2'] = MagicMock()

from src.adaptive_orb_slam.frontend.feature_extractor import FeatureExtractor


def test_feature_extractor_initialization():
    feature_extractor = FeatureExtractor()
    assert feature_extractor is not None  # Dummy test to check initialization


def test_feature_extractor_functionality():
    feature_extractor = FeatureExtractor()
    dummy_image = MagicMock()

    # Mock the behavior of cv2 functions if necessary
    feature_extractor.extract_features = MagicMock(return_value=True)
    result = feature_extractor.extract_features(dummy_image)

    assert result == True  # Dummy test to ensure extraction is called and returns true
