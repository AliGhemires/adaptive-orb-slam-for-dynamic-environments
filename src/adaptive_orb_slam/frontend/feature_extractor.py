import cv2
import numpy as np
from typing import List, Tuple, Optional

class FeatureExtractor:
    """
    A class to extract features from images using ORB (Oriented FAST and Rotated BRIEF).
    
    Features are used to identify points of interest in visual SLAM applications.
    """

    def __init__(self, n_features: int = 1000):
        """
        Initializes the FeatureExtractor.
        
        Args:
            n_features (int): The maximum number of features to extract. Defaults to 1000.
        """
        assert n_features > 0, "Number of features must be positive"
        self.n_features = n_features
        self.orb = cv2.ORB_create(n_features)

    def extract_features(self, image: np.ndarray) -> Tuple[Optional[List[cv2.KeyPoint]], Optional[np.ndarray]]:
        """
        Extracts ORB keypoints and descriptors from an input image.
        
        Args:
            image (np.ndarray): Input image from which to extract features.
            Must be a grayscale image.

        Returns:
            Tuple[Optional[List[cv2.KeyPoint]], Optional[np.ndarray]]: A tuple containing a list of keypoints and an array of their corresponding descriptors.
            Returns (None, None) if extraction fails or if no keypoints are detected.
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array.")
        
        if image.ndim != 2:
            raise ValueError("Input image must be a grayscale image.")
        
        # Extract features
        keypoints, descriptors = self.orb.detectAndCompute(image, None)

        if keypoints is None or len(keypoints) == 0:
            return None, None
        
        # Return results
        return keypoints, descriptors

    def draw_keypoints(self, image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """
        Draws detected keypoints on the image.
        
        Args:
            image (np.ndarray): The image on which to draw the keypoints.
            keypoints (List[cv2.KeyPoint]): List of keypoints detected from the previous extraction.

        Returns:
            np.ndarray: The input image with keypoints drawn on it.
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array.")

        if keypoints is None:
            raise ValueError("Keypoints list is None.")

        if not isinstance(keypoints, list):
            raise ValueError("Keypoints must be a list.")

        # Draw keypoints
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return image_with_keypoints

# Example of using the FeatureExtractor class
if __name__ == "__main__":
    extractor = FeatureExtractor(n_features=1500)  # Create an extractor with 1500 features.

    image = cv2.imread('path_to_image.jpg', cv2.IMREAD_GRAYSCALE)  # Load a grayscale image.
    if image is None:
        raise FileNotFoundError("Could not read the image file.")

    keypoints, descriptors = extractor.extract_features(image)  # Extract features.
    if keypoints is not None:
        image_with_keypoints = extractor.draw_keypoints(image, keypoints)  # Draw keypoints.
        cv2.imshow('Keypoints', image_with_keypoints)  # Show image with keypoints.
        cv2.waitKey(0)  # Wait for a keypress to close the window.
        cv2.destroyAllWindows()  # Close the OpenCV window.
    else:
        print("No keypoints detected.")
