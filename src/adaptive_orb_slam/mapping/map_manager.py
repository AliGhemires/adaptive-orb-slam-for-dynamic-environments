import numpy as np
from typing import List, Dict, Any, Optional

class MapManager:
    """
    A class that manages the mapping operation in an ORB-SLAM system.
    It handles the storage and updates of the map as new observations are made.

    Attributes:
    - keyframes: A list of keyframe data points in the map.
    - map_points: A dictionary mapping point IDs to their 3D coordinates.
    - map_id_counter: Counter to assign unique IDs to map points.

    Methods:
    - add_keyframe: Adds a new keyframe to the map.
    - add_map_point: Adds a new 3D point to the map.
    - remove_map_point: Removes a 3D point from the map.
    - get_map_points: Retrieves all 3D map points.
    - get_keyframes: Retrieves all keyframes.
    """  
    def __init__(self) -> None:
        self.keyframes: List[Dict[str, Any]] = []
        self.map_points: Dict[int, np.ndarray] = {}
        self.map_id_counter: int = 0

    def add_keyframe(self, keyframe_data: Dict[str, Any]) -> None:
        """
        Add a new keyframe to the map.
        
        Parameters:
        - keyframe_data: A dictionary containing keyframe information, such as:
            {'id': int, 'descriptor': np.ndarray}

        Throws:
        - ValueError: If keyframe_data lacks required fields or contains invalid data.
        """
        if 'id' not in keyframe_data or not isinstance(keyframe_data['descriptor'], np.ndarray):
            raise ValueError("Keyframe data must contain an 'id' and a valid 'descriptor'.")
        self.keyframes.append(keyframe_data.copy())  # copy to prevent external mutation

    def add_map_point(self, point: np.ndarray) -> int:
        """
        Add a new 3D point to the map and return its unique ID.

        Parameters:
        - point: A 3D point as a NumPy array of shape (3,).  

        Returns:
        - int: The ID of the newly added point.

        Throws:
        - ValueError: If point is not a 3D vector.
        """
        if point.shape != (3,):
            raise ValueError("Point must be a 3D vector of shape (3,).")
        point_id = self.map_id_counter
        self.map_points[point_id] = point.copy()  # store a copy to ensure integrity
        self.map_id_counter += 1 
        return point_id

    def remove_map_point(self, point_id: int) -> None:
        """
        Remove a map point by its ID.

        Parameters:
        - point_id: The ID of the point to remove.

        Throws:
        - KeyError: If the point ID does not exist.
        """
        try:
            del self.map_points[point_id]
        except KeyError:
            raise KeyError(f"Map point with ID {point_id} does not exist.")

    def get_map_points(self) -> Dict[int, np.ndarray]:
        """
        Retrieves all 3D map points.
        
        Returns:
        - dict: A dictionary of map point IDs to their 3D coordinates.
        """  
        return {point_id: point.copy() for point_id, point in self.map_points.items()}  # return deep copy to prevent mutations

    def get_keyframes(self) -> List[Dict[str, Any]]:
        """
        Retrieves all keyframes in the current map.
        
        Returns:
        - list: A list of keyframe data dictionaries.
        """  
        return [kf.copy() for kf in self.keyframes]  # return deep copy for safety

# Example usage
if __name__ == "__main__":
    manager = MapManager()
    kf_data = {'id': 1, 'descriptor': np.random.rand(128)}
    manager.add_keyframe(kf_data)
    point_id = manager.add_map_point(np.array([1.0, 2.0, 3.0]))
    print(f"Added map point with ID: {point_id}")
    print(f"Current keyframes: {manager.get_keyframes()}")
    print(f"Current map points: {manager.get_map_points()}")
