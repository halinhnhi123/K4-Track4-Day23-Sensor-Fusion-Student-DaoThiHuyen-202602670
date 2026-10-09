"""Sensor adapters for vehicle-frame states and calibrated pinhole observations.

Waymo camera axes are forward, left, up. The horizontal pixel interval [0, width]
therefore maps to angles atan((c_i - width) / f_i) through atan(c_i / f_i).
The camera Jacobian follows the chain rule: projection derivative times rotation.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab import tracking_params as params

Matrix = np.matrix | np.ndarray
MIN_CAMERA_DEPTH = 1e-6


def _noise_covariance(name: str) -> np.matrix:
    """Construct independent observation noise in the sensor's native units."""
    scales = {
        "lidar": (params.sigma_lidar_x, params.sigma_lidar_y, params.sigma_lidar_z),
        "camera": (params.sigma_cam_i, params.sigma_cam_j),
    }
    return np.asmatrix(np.diag(np.square(scales[name])))


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a small calibration transform without calling native LAPACK."""
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Camera extrinsic transform must be a finite 4x4 matrix")

    augmented = [
        [float(transform[row, column]) for column in range(4)]
        + [float(row == column) for column in range(4)]
        for row in range(4)
    ]
    for pivot_column in range(4):
        pivot_row = max(
            range(pivot_column, 4),
            key=lambda row: abs(augmented[row][pivot_column]),
        )
        pivot = augmented[pivot_row][pivot_column]
        if pivot == 0:
            raise ValueError("Camera extrinsic transform must be invertible")
        augmented[pivot_column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot_column],
        )

        pivot = augmented[pivot_column][pivot_column]
        augmented[pivot_column] = [value / pivot for value in augmented[pivot_column]]
        for row in range(4):
            if row == pivot_column:
                continue
            factor = augmented[row][pivot_column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[pivot_column]
                )
            ]
    return np.asarray([row[4:] for row in augmented], dtype=float)


def _matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply small calibration matrices without dispatching to native BLAS."""
    return np.sum(left[:, :, None] * right[None, :, :], axis=1)


class Sensor:
    """Adapt lidar positions or calibrated camera pixels to a six-state EKF.

    Args:
        name: Either ``lidar`` or ``camera``.
        calib: Waymo camera calibration, including image width; unused for lidar.
        camera_fusion: Student module implementing visibility and camera projection.

    Raises:
        ValueError: If the sensor type or camera calibration is invalid.
    """

    depth_epsilon = MIN_CAMERA_DEPTH

    def __init__(self, name: str, calib: Any, camera_fusion: Any) -> None:
        dimensions = {"lidar": 3, "camera": 2}
        if name not in dimensions:
            raise ValueError(f"Unsupported tracking sensor: {name!r}")
        self.name = name
        self.dim_meas = dimensions[name]
        self._cam = camera_fusion
        transform = np.eye(4, dtype=float)
        angular_bounds = (-np.pi / 2, np.pi / 2)
        if name == "camera":
            intrinsics = np.asarray(calib.intrinsic, dtype=float)
            if intrinsics.size < 4:
                raise ValueError("Camera calibration needs fx, fy, cx, cy")
            self.f_i, self.f_j, self.c_i, self.c_j = intrinsics[:4]
            self.image_width = float(calib.width)
            if (
                not np.isfinite(intrinsics[:4]).all()
                or min(self.f_i, self.f_j, self.image_width) <= 0
                or not np.isfinite(self.image_width)
            ):
                raise ValueError("Camera focal lengths and image width must be positive")
            transform = np.asarray(calib.extrinsic.transform, dtype=float).reshape(4, 4)
            angular_bounds = np.arctan(
                (self.c_i - np.array([self.image_width, 0.0])) / self.f_i
            )
        self.sens_to_veh = np.asmatrix(transform)
        self.veh_to_sens = np.asmatrix(_invert_transform(transform))
        self.fov = tuple(float(bound) for bound in angular_bounds)

    def _position_in_sensor(self, x: Matrix) -> np.ndarray:
        position = np.asarray(x, dtype=float).reshape(-1)[:3]
        if not np.isfinite(position).all():
            return np.full(3, np.nan)
        transform = np.asarray(self.veh_to_sens)
        transformed = _matmul(
            transform[:3, :3], position.reshape(-1, 1)
        ).reshape(-1)
        return transformed + transform[:3, 3]

    def _camera_position(self, x: Matrix) -> np.ndarray:
        position = self._position_in_sensor(x)
        if not np.isfinite(position).all() or position[0] <= self.depth_epsilon:
            raise ValueError(
                "Camera projection needs finite coordinates and positive depth "
                f"> {self.depth_epsilon:g}; sensor position={position.tolist()}"
            )
        return position

    def in_fov(self, x: Matrix) -> bool:
        """Return whether the student visibility model admits this state.

        Args:
            x: Vehicle-frame state vector.

        Returns:
            Visibility, with nonfinite or nonpositive camera depth always rejected.
        """
        if self.name == "camera":
            position = self._position_in_sensor(x)
            if not np.isfinite(position).all() or position[0] <= self.depth_epsilon:
                return False
        return bool(self._cam.is_in_field_of_view(x, self))

    def get_hx(self, x: Matrix) -> Matrix:
        """Predict the position or pixel observation.

        Args:
            x: Vehicle-frame state vector.

        Returns:
            A column vector in measurement coordinates.

        Raises:
            ValueError: If camera coordinates are nonfinite or depth is not positive.
        """
        if self.name == "camera":
            self._camera_position(x)
            return self._cam.camera_measurement_prediction(x, self)
        return np.asmatrix(self._position_in_sensor(x)).T

    def get_H(self, x: Matrix) -> np.matrix:
        """Differentiate the observation with respect to position and velocity.

        Args:
            x: Vehicle-frame state vector.

        Returns:
            Measurement-by-state Jacobian, with zero velocity columns.

        Raises:
            ValueError: If camera coordinates are nonfinite or depth is not positive.
        """
        rotation = np.asarray(self.veh_to_sens)[:3, :3]
        derivative = rotation
        if self.name == "camera":
            depth, left, up = self._camera_position(x)
            projection = np.array([
                [self.f_i * left / depth**2, -self.f_i / depth, 0.0],
                [self.f_j * up / depth**2, 0.0, -self.f_j / depth],
            ])
            derivative = _matmul(projection, rotation)
        return np.asmatrix(
            np.pad(derivative, ((0, 0), (0, params.dim_state - 3)))
        )

    def generate_measurement(
        self, num_frame: int, z: Sequence[float], meas_list: list[Any]
    ) -> list[Any]:
        """Append an observation at the zero-based dataset frame timestamp.

        Args:
            num_frame: Nonnegative dataset frame index; frame zero has time zero.
            z: Lidar ``[x, y, z, height, width, length, yaw]`` or camera ``[u, v]``.
            meas_list: Observation list to extend in place.

        Returns:
            The supplied list, extended by one observation.
        """
        payload = {}
        if self.name == "camera":
            built = self._cam.build_camera_measurement(z, self)
            payload = {"z_mat": built["z"], "R": built["R"]}
        meas_list.append(Measurement(num_frame, z, self, **payload))
        return meas_list


class Measurement:
    """Observation values, covariance, time, and optional lidar box dimensions.

    Args:
        num_frame: Nonnegative, zero-based dataset frame index.
        z: Native observation vector, followed by lidar box dimensions and yaw.
        sensor: Adapter that generated this observation.
        z_mat: Optional prebuilt column vector from the camera exercise.
        R: Optional covariance from the camera exercise.

    Raises:
        ValueError: If the frame index is negative.
    """

    def __init__(
        self,
        num_frame: int,
        z: Sequence[float],
        sensor: Sensor,
        z_mat: Matrix | None = None,
        R: Matrix | None = None,
    ) -> None:
        if num_frame < 0:
            raise ValueError(f"Measurement frame index must be nonnegative: {num_frame}")
        self.sensor = sensor
        self.t = float(num_frame * params.dt)
        values = np.asarray(z if z_mat is None else z_mat, dtype=float).reshape(-1)
        self.z = np.asmatrix(values[:sensor.dim_meas]).T
        self.R = _noise_covariance(sensor.name) if R is None else np.asmatrix(R)
        if sensor.name == "lidar":
            self.height, self.width, self.length, self.yaw = map(float, values[3:7])
