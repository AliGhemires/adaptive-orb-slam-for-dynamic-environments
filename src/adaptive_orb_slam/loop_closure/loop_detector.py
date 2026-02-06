import numpy as np
from typing import List, Tuple, Optional, Dict

class LoopDetector:
    """
    A class to detect loop closures in Visual SLAM using feature descriptors.

    Attributes:
        descriptors (List[np.ndarray]): Feature descriptors of past frames.
        threshold (float): A distance threshold for loop closure detection.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        """ 
        Initializes the LoopDetector with a distance threshold.

        Parameters:
            threshold (float): The threshold for considering a loop closure.
        """
        self.descriptors: List[np.ndarray] = []  # List to store feature descriptors
        self.threshold: float = threshold

    def add_descriptors(self, descriptor: np.ndarray) -> None:
        """ 
        Adds a feature descriptor of a new frame to the list.

        Parameters:
            descriptor (np.ndarray): The feature descriptor to add.
        Raises:
            ValueError: If the descriptor is not a valid numpy array.
        """
        if not isinstance(descriptor, np.ndarray):
            raise ValueError("Descriptor must be a numpy array.")
        if descriptor.ndim != 1:
            raise ValueError("Descriptor must be a 1-dimensional numpy array.")
        self.descriptors.append(descriptor)

    def detect_loop(self, current_descriptor: np.ndarray) -> Optional[int]:
        """
        Detects a loop closure by matching the current descriptor with past descriptors.

        Parameters:
            current_descriptor (np.ndarray): The feature descriptor of the current frame.

        Returns:
            Optional[int]: The index of the detected loop closure in the descriptors list,
            or None if no loop is detected.
        Raises:
            ValueError: If the current descriptor is not a valid numpy array.
        """
        if not isinstance(current_descriptor, np.ndarray):
            raise ValueError("Current descriptor must be a numpy array.")
        if current_descriptor.ndim != 1:
            raise ValueError("Current descriptor must be a 1-dimensional numpy array.")

        # Match the current descriptor against all stored descriptors
        best_match_index: Optional[int] = None
        best_distance: float = float('inf')
        
        for index, past_descriptor in enumerate(self.descriptors):
            distance = self.compute_distance(current_descriptor, past_descriptor)
            # Ensure numerical stability by checking for non-finite values
            if not np.isfinite(distance):
                continue
            if distance < best_distance:
                best_distance = distance
                best_match_index = index

        if best_distance < self.threshold:
            return best_match_index

        return None

    def compute_distance(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """
        Computes the distance between two feature descriptors.

        Parameters:
            desc1 (np.ndarray): The first feature descriptor.
            desc2 (np.ndarray): The second feature descriptor.

        Returns:
            float: The Euclidean distance between the two descriptors.
        Raises:
            AssertionError: If descriptors do not have the same shape or are not 1-D.
        """
        if desc1.shape != desc2.shape or desc1.ndim != 1:
            raise AssertionError("Descriptors must be 1-dimensional and of the same shape.")
        distance = np.linalg.norm(desc1 - desc2)
        return distance

    def get_last_descriptor(self) -> Optional[np.ndarray]:
        """
        Returns the last feature descriptor added.

        Returns:
            Optional[np.ndarray]: The last descriptor or None if no descriptors exist.
        """
        return self.descriptors[-1] if self.descriptors else None

    def clear_descriptors(self) -> None:
        """
        Clears the stored feature descriptors.
        """
        self.descriptors.clear()
