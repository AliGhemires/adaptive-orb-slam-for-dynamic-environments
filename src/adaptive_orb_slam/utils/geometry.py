import numpy as np
from typing import List, Tuple, Optional


class Geometry:
    @staticmethod
    def compute_homogeneous_coordinates(points: List[Tuple[float, float]]) -> np.ndarray:
        """Convert 2D points to homogeneous coordinates.

        Args:
            points (List[Tuple[float, float]]): List of 2D points as tuples (x, y).

        Returns:
            np.ndarray: Array of points in homogeneous coordinates.

        Raises:
            ValueError: If the input list of points is empty.
        """
        # Validate input
        if not points:
            raise ValueError('Input points list cannot be empty.')

        return np.vstack([np.array(points).T, np.ones(len(points))])

    @staticmethod
    def compute_affine_transformation(src_points: List[Tuple[float, float]],
                                       dst_points: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """Compute the affine transformation matrix that maps src_points to dst_points.

        Args:
            src_points (List[Tuple[float, float]]): Source points.
            dst_points (List[Tuple[float, float]]): Destination points.

        Returns:
            Optional[np.ndarray]: Affine transformation matrix of shape (3, 3) or None if the computation fails.

        Raises:
            ValueError: If fewer than 3 pairs of corresponding points are provided.
        """
        # Validate input
        if len(src_points) != len(dst_points) or len(src_points) < 3:
            raise ValueError('At least 3 pairs of corresponding points are needed for affine transformation.')

        src = Geometry.compute_homogeneous_coordinates(src_points)
        dst = Geometry.compute_homogeneous_coordinates(dst_points)

        # Prepare matrices for solving
        A = np.vstack((src[:2].T, np.ones(len(src_points)))).T
        B = dst[:2].T

        # Solve for the transformation matrix using least squares
        A_solution, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        # Construct the affine transformation matrix
        transform_matrix = np.eye(3)
        transform_matrix[:2, :] = A_solution.T

        return transform_matrix

    @staticmethod
    def apply_transformation(points: List[Tuple[float, float]],
                             transformation: np.ndarray) -> np.ndarray:
        """Apply a transformation matrix to a list of points.

        Args:
            points (List[Tuple[float, float]]): List of points to transform.
            transformation (np.ndarray): Transformation matrix.

        Returns:
            np.ndarray: Transformed points.

        Raises:
            ValueError: If transformation matrix is not 3x3 or points list is empty.
        """
        if transformation.shape != (3, 3):
            raise ValueError('Transformation matrix must be 3x3.')

        if not points:
            raise ValueError('Input points list cannot be empty.')

        homogeneous_points = Geometry.compute_homogeneous_coordinates(points)

        # Apply the transformation
        transformed_points = transformation @ homogeneous_points

        # Convert back to Cartesian coordinates
        return (transformed_points[:-1] / transformed_points[-1]).T

    @staticmethod
    def compute_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two 2D points.

        Args:
            point1 (Tuple[float, float]): First point.
            point2 (Tuple[float, float]): Second point.

        Returns:
            float: Distance between point1 and point2.
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))

    @staticmethod
    def centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Compute the centroid of a list of 2D points.

        Args:
            points (List[Tuple[float, float]]): List of 2D points.

        Returns:
            Tuple[float, float]: Centroid point (x, y).

        Raises:
            ValueError: If the input list of points is empty.
        """
        if not points:
            raise ValueError('Input points list cannot be empty.')

        x_coords, y_coords = zip(*points)
        return (np.mean(x_coords), np.mean(y_coords))


# Example usage of the Geometry class
if __name__ == '__main__':
    points_a = [(0, 0), (1, 0), (0, 1)]
    points_b = [(1, 1), (2, 1), (1, 2)]
    try:
        transformation_matrix = Geometry.compute_affine_transformation(points_a, points_b)
        transformed_points = Geometry.apply_transformation(points_a, transformation_matrix)
        print('Affine Transformation Matrix:')
        print(transformation_matrix)
        print('Transformed Points:')
        print(transformed_points)
    except ValueError as ve:
        print(f'ValueError: {ve}')
