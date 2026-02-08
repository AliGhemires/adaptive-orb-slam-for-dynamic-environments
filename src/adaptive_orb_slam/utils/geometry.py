import numpy as np
from typing import List, Tuple, Optional, Union


class Geometry:
    """
    A class for performing various geometric operations required in visual SLAM,
    including calculations for transformations, distances, and projections.
    """

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.
        
        Args:
            v (np.ndarray): The input vector to normalize.

        Returns:
            np.ndarray: The normalized vector.
            
        Raises:
            ValueError: If the vector is zero or has an invalid shape.
        """
        if not isinstance(v, np.ndarray) or v.ndim != 1:
            raise ValueError("Input must be a one-dimensional np.ndarray")
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector")
        return v / norm

    @staticmethod
    def distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            point1 (np.ndarray): First point in n-dimensional space.
            point2 (np.ndarray): Second point in n-dimensional space.

        Returns:
            float: The Euclidean distance between the points.
            
        Raises:
            ValueError: If the inputs do not have the same shape or are not valid arrays.
        """
        if not (isinstance(point1, np.ndarray) and isinstance(point2, np.ndarray)):
            raise ValueError("Inputs must be np.ndarray")
        if point1.shape != point2.shape:
            raise ValueError("Points must have the same dimensions")
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def transform_point(point: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
        """
        Transform a point using a transformation matrix (4x4).
        
        Args:
            point (np.ndarray): The point as a homogeneous coordinate [x, y, z, 1].
            transformation_matrix (np.ndarray): The transformation matrix (4x4).

        Returns:
            np.ndarray: The transformed point as a homogeneous coordinate.
            
        Raises:
            ValueError: If the point is not in homogeneous coordinates or if the dimension mismatch occurs.
        """
        if not isinstance(point, np.ndarray) or not isinstance(transformation_matrix, np.ndarray):
            raise ValueError("Both arguments must be np.ndarray")
        if point.shape != (4,):
            raise ValueError("Point must be a 1D array in homogeneous coordinates")
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")
        return transformation_matrix @ point

    @staticmethod
    def project_point(point: np.ndarray, camera_matrix: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Project a 3D point onto a 2D image plane using a camera matrix.
        
        Args:
            point (np.ndarray): The 3D point to be projected.
            camera_matrix (np.ndarray): The intrinsic camera matrix (3x3).

        Returns:
            Optional[Tuple[float, float]]: The 2D coordinates on the image plane (u, v),
            or None if the point is at infinity.
            
        Raises:
            ValueError: If the point does not have the correct dimensionality.
        """
        if not isinstance(point, np.ndarray) or not isinstance(camera_matrix, np.ndarray):
            raise ValueError("Both arguments must be np.ndarray")
        if point.shape != (3,):
            raise ValueError("Point must be a 3D vector")
        if camera_matrix.shape != (3, 3):
            raise ValueError("Camera matrix must be 3x3")
        homogeneous_point = np.append(point, 1)
        projected_point = camera_matrix @ homogeneous_point
        # Normalize by the third coordinate (homogeneous coordinate)
        if projected_point[2] == 0:
            return None  # Point is at infinity
        return (projected_point[0] / projected_point[2], projected_point[1] / projected_point[2])

    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate the angle in radians between two vectors.
        
        Args:
            v1 (np.ndarray): First vector.
            v2 (np.ndarray): Second vector.

        Returns:
            float: Angle in radians between the two vectors.
            
        Raises:
            ValueError: If either vector is zero, not valid, or if they don't have the same dimension.
        """
        if not (isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)):
            raise ValueError("Inputs must be np.ndarray")
        if v1.shape != v2.shape:
            raise ValueError("Vectors must have the same dimensions")
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            raise ValueError("Vectors must be non-zero")
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.arccos(cos_angle)

    @staticmethod
    def compute_plane_normal(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """
        Compute the normal vector of a plane defined by three points.
        
        Args:
            p1 (np.ndarray): First point of the plane.
            p2 (np.ndarray): Second point of the plane.
            p3 (np.ndarray): Third point of the plane.

        Returns:
            np.ndarray: The normal vector of the plane.

        Raises:
            ValueError: If the points are not in 3D space or they are collinear.
        """
        if not (isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray) and isinstance(p3, np.ndarray)):
            raise ValueError("All inputs must be np.ndarray")
        if p1.shape != (3,) or p2.shape != (3,) or p3.shape != (3,):
            raise ValueError("All points must be 1D arrays in 3D space")
        # Create vectors from points
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) == 0:
            raise ValueError("Points must not be collinear")
        return Geometry.normalize(normal)

# Example of testing geometric utility functions using assert
if __name__ == '__main__':
    # Testing normalization
    vec = np.array([3.0, 4.0])
    assert np.allclose(Geometry.normalize(vec), np.array([0.6, 0.8])), "Normalization test failed"
    
    # Testing distance calculation
    p1 = np.array([1.0, 2.0])
    p2 = np.array([4.0, 6.0])
    assert np.isclose(Geometry.distance(p1, p2), 5.0), "Distance test failed"
    
    # Testing point transformation
    transformation_matrix = np.array([[1, 0, 0, 1],
                                       [0, 1, 0, 2],
                                       [0, 0, 1, 3],
                                       [0, 0, 0, 1]])
    point = np.array([1, 1, 1, 1])
    transformed = Geometry.transform_point(point, transformation_matrix)
    assert np.allclose(transformed, np.array([2, 3, 4, 1])), "Transformation test failed"
    
    print("All tests passed!")
