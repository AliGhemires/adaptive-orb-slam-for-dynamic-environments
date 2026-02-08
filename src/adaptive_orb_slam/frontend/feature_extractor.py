import cv2
import numpy as np
from typing import List, Tuple

class FeatureExtractor:
    """
    A class to handle feature extraction in a visual SLAM system using ORB.
    This class encapsulates the ORB feature extractor and provides methods for keypoint detection
    and descriptor computation.
    """

    def __init__(self, n_features: int = 500) -> None:
        """
        Initializes the FeatureExtractor with optional parameters.
        
        :param n_features: The maximum number of features to retain (default is 500).
        """
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detects keypoints and computes their descriptors in the provided image.
        
        :param image: Input image from which to extract features.
        :return: A tuple containing a list of detected keypoints and their descriptors.
        
        :raises ValueError: If the input image is not a valid grayscale or color image.
        """
        # Validate the input image
        self._validate_image(image)

        # Convert to grayscale if the image is a color image
        if image.ndim == 3 and image.shape[2] == 3:  # Ensure image is in standard BGR format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        if keypoints is None or descriptors is None:
            raise RuntimeError("Failed to detect keypoints or compute descriptors.")

        return keypoints, descriptors

    def _validate_image(self, image: np.ndarray) -> None:
        """
        Validates if the input image is valid (non-empty and correct dimensions).
        
        :param image: The image to be validated.
        
        :raises ValueError: If the image is empty or not 2D or 3D.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array.")
        if image.size == 0:
            raise ValueError("Input image cannot be empty.")
        if image.ndim not in [2, 3]:
            raise ValueError("Input image must be either grayscale (2D) or color (3D) with 3 channels.")

    def draw_keypoints(self, image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """
        Draws the detected keypoints on the image for visualization.
        
        :param image: Input image where keypoints will be drawn.
        :param keypoints: List of detected keypoints to draw.
        :return: Image with keypoints drawn.
        """
        # Validate inputs
        self._validate_image(image)
        if keypoints is None:
            raise ValueError("Keypoints should not be None.")

        # Draw keypoints
        return cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        A high-level method to extract features and optionally visualize them.
        
        :param image: Input image for feature extraction.
        :return: A tuple containing keypoints and descriptors.
        
        :raises RuntimeError: If keypoints or descriptors cannot be extracted.
        """
        keypoints, descriptors = self.detect_and_compute(image)
        # Optional: Draw keypoints for debugging
        annotated_image = self.draw_keypoints(image, keypoints)
        return keypoints, descriptors

# Example usage (uncomment to use in testing or actual implementation):
# if __name__ == '__main__':
#     img = cv2.imread('path_to_image.jpg')
#     feature_extractor = FeatureExtractor()
#     keypoints, descriptors = feature_extractor.extract_features(img)
#     cv2.imshow('Keypoints', feature_extractor.draw_keypoints(img, keypoints))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
