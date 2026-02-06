import numpy as np
from typing import List, Tuple, Optional

class Optimizer:
    """
    A class to perform optimization on the map built during SLAM.

    This class implements essential optimization routines to refine the map and improve the camera pose estimates
    using non-linear optimization techniques such as bundle adjustment.
    """

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initializes the optimizer with given parameters.

        Args:
            max_iterations (int): Maximum number of iterations for optimization. Default is 100.
            tolerance (float): Convergence tolerance. Default is 1e-6.
        """
        if max_iterations <= 0:
            raise ValueError("Max iterations must be a positive integer.")
        if tolerance <= 0:
            raise ValueError("Tolerance must be a positive float.")
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self, initial_pose: np.ndarray, observations: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Perform optimization on the provided pose estimates based on observations.

        Args:
            initial_pose (np.ndarray): Initial camera pose.
            observations (List[Tuple[np.ndarray, np.ndarray]]): List of feature position observations in both the
                world coordinates and the corresponding camera coordinates.

        Returns:
            np.ndarray: The optimized camera pose.

        Raises:
            ValueError: If the input data is invalid or not as expected.
        """

        # Validate input data
        if not isinstance(initial_pose, np.ndarray) or initial_pose.shape != (3, 4):
            raise ValueError(f"Expected initial_pose as (3, 4) ndarray, got {type(initial_pose)} with shape {initial_pose.shape}")
        if not isinstance(observations, list) or not all(isinstance(obs, tuple) and len(obs) == 2 and all(isinstance(arr, np.ndarray) for arr in obs) for obs in observations):
            raise ValueError("Observations must be a list of tuples, each containing two numpy arrays.")

        # Ensure no zero-sized inputs
        if len(observations) == 0:
            raise ValueError("Observations list cannot be empty.")

        current_pose = initial_pose.copy()
        for iteration in range(self.max_iterations):
            residuals = self._calculate_residuals(current_pose, observations)
            jacobian = self._calculate_jacobian(current_pose, observations)

            # Solve the linear system
            try:
                pose_update = self._linear_solve(jacobian, residuals)
            except np.linalg.LinAlgError:
                raise RuntimeError("Linear solver failed due to singular matrix.")

            # Update the pose
            current_pose -= pose_update

            # Check for convergence
            if np.linalg.norm(pose_update) < self.tolerance:
                break

        return current_pose

    def _calculate_residuals(self, pose: np.ndarray, observations: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Calculate the residuals between the observed feature positions and the projected positions based on the pose.

        Args:
            pose (np.ndarray): Current camera pose.
            observations (List[Tuple[np.ndarray, np.ndarray]]): List of feature observations.

        Returns:
            np.ndarray: The residuals (differences between predicted and observed feature positions).
        """
        residuals = np.zeros((len(observations), 2))

        for i, (world_point, observed_point) in enumerate(observations):
            if world_point.shape[0] != 3 or observed_point.shape[0] != 2:
                raise ValueError("World point must be 3D and observed point must be 2D.")
            projected_point = self._project_point(world_point, pose)
            residuals[i, :] = observed_point - projected_point

        return residuals.flatten()

    def _calculate_jacobian(self, pose: np.ndarray, observations: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Compute the Jacobian matrix for the optimization process.

        Args:
            pose (np.ndarray): Current camera pose.
            observations (List[Tuple[np.ndarray, np.ndarray]]): List of feature observations.

        Returns:
            np.ndarray: The Jacobian matrix.
        """
        num_observations = len(observations)
        jacobian = np.zeros((num_observations * 2, 6))

        for i, (world_point, _) in enumerate(observations):
            jacobian[2*i:2*i+2, :] = self._compute_jacobian_row(world_point, pose)

        return jacobian

    def _linear_solve(self, jacobian: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """
        Solve the linear system using the normal equations to find the pose update.

        Args:
            jacobian (np.ndarray): Jacobian matrix.
            residuals (np.ndarray): Current residuals.

        Returns:
            np.ndarray: The update to the pose parameters.

        Raises:
            RuntimeError: If the matrix is singular.
        """
        J_T = jacobian.T
        normal_matrix = J_T @ jacobian
        rhs = -J_T @ residuals

        if np.linalg.cond(normal_matrix) < 1 / np.finfo(normal_matrix.dtype).eps:
            delta = np.linalg.solve(normal_matrix, rhs)
        else:
            raise RuntimeError("Normal matrix is ill-conditioned.")

        return delta

    def _project_point(self, point: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Project a 3D point to 2D using the camera pose.

        Args:
            point (np.ndarray): 3D point in world coordinates.
            pose (np.ndarray): Current camera pose.

        Returns:
            np.ndarray: The projected 2D point in image coordinates.
        """
        # Pinhole camera model mocked projection
        # Replace with actual projection computation
        projected = point[:2] / point[2]
        return projected

    def _compute_jacobian_row(self, world_point: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian row for a single observed point.

        Args:
            world_point (np.ndarray): The 3D point in world coordinates.
            pose (np.ndarray): The current camera pose.

        Returns:
            np.ndarray: The Jacobian row corresponding to the given world point.
        """
        # Mock Jacobian; replace with analytical derivative
        return np.eye(6)  # Replace with actual calculation.

# Example usage
# optimizer = Optimizer()
# optimized_pose = optimizer.optimize(initial_pose, observations)  # where observations are given.
