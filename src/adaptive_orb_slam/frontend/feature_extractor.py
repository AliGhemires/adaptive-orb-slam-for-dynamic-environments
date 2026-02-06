import numpy as np
import cv2
from typing import List, Tuple

class FeatureExtractor:
    """
    A class to extract and match features from images using ORB (Oriented FAST and Rotated BRIEF) algorithm.

    This implementation focuses on providing a clean interface for feature extraction, ensuring numerical stability
    and robustness against variations in input images.
    """

    def __init__(self, n_features: int = 500) -> None:
        """
        Initialize the FeatureExtractor with a specified number of features to extract.

        Args:
            n_features (int): The maximum number of features to extract from an image.
        """
        if n_features <= 0:
            raise ValueError("Number of features must be positive")
        self.n_features = n_features
        self.orb = cv2.ORB_create(nfeatures=self.n_features)

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract ORB features from the provided image.

        Args:
            image (np.ndarray): Input image from which to extract features (must be grayscale).

        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray]: A tuple containing:
                - keypoints (List[cv2.KeyPoint]): Detected keypoints.
                - descriptors (np.ndarray): Feature descriptors corresponding to the keypoints.
        """
        if image is None:
            raise ValueError("Input image cannot be None")

        if image.ndim != 2:
            raise ValueError("Input image must be a grayscale image")

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image, None)

        if not keypoints:
            raise RuntimeError("No keypoints found. Ensure the image has salient features.")
        if descriptors is None or descriptors.shape[0] != len(keypoints):
            raise RuntimeError("Descriptors were not found or misaligned with keypoints.")

        return keypoints, descriptors

    def match_features(self,
                      descriptors1: np.ndarray,
                      descriptors2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two sets of descriptors using the BFMatcher (Brute Force Matcher).

        Args:
            descriptors1 (np.ndarray): Feature descriptors from the first image.
            descriptors2 (np.ndarray): Feature descriptors from the second image.

        Returns:
            List[cv2.DMatch]: A list of matches between the two descriptors.
        """
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Descriptors must not be None")

        if descriptors1.shape[1] != descriptors2.shape[1]:
            raise ValueError("Descriptor dimensions do not match")

        # Initialize BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches based on distance
        matches.sort(key=lambda x: x.distance)

        return matches

    def draw_matches(self, image1: np.ndarray,
                     image2: np.ndarray,
                     keypoints1: List[cv2.KeyPoint],
                     keypoints2: List[cv2.KeyPoint],
                     matches: List[cv2.DMatch]) -> np.ndarray:
        """
        Draw matches between two images with a visual representation of the keypoints and matches.

        Args:
            image1 (np.ndarray): The first image.
            image2 (np.ndarray): The second image.
            keypoints1 (List[cv2.KeyPoint]): Keypoints from the first image.
            keypoints2 (List[cv2.KeyPoint]): Keypoints from the second image.
            matches (List[cv2.DMatch]): Matches between keypoints.

        Returns:
            np.ndarray: Image with matches drawn.
        """
        if len(image1.shape) != 2 or len(image2.shape) != 2:
            raise ValueError("Both images must be grayscale.")

        if not keypoints1 or not keypoints2:
            raise ValueError("Keypoints list cannot be empty.")

        if not matches:
            raise ValueError("Matches list cannot be empty.")

        # Draw matches
        matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return matched_image

# Note: Ensure that OpenCV (cv2) is installed and properly imported in the environment for this code to execute correctly.
