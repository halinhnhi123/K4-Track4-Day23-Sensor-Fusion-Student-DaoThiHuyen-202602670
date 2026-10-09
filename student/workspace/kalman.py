"""Extended Kalman filter helpers for 6D constant-velocity motion.

Part E supplies prediction and correction for docs/HUONG_DAN_KY_THUAT.md §2.
Read the shared time step and process-noise settings with get_tracking_params().
"""

from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.ndarray

params = get_tracking_params()


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    """Multiply small matrices without dispatching to the platform BLAS."""
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("Matrix multiplication requires two 2D arrays")
    if left_array.shape[1] != right_array.shape[0]:
        raise ValueError(
            f"Incompatible matrix shapes: {left_array.shape} and {right_array.shape}"
        )
    return np.sum(left_array[:, :, None] * right_array[None, :, :], axis=1)


def _inverse(matrix: Matrix) -> Matrix:
    """Invert a small square matrix with pivoted Gauss-Jordan elimination."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Matrix inverse requires a square 2D array")

    size = values.shape[0]
    augmented = [
        [float(values[row, column]) for column in range(size)]
        + [float(row == column) for column in range(size)]
        for row in range(size)
    ]

    for pivot_column in range(size):
        pivot_row = max(
            range(pivot_column, size),
            key=lambda row: abs(augmented[row][pivot_column]),
        )
        pivot = augmented[pivot_row][pivot_column]
        if pivot == 0:
            raise np.linalg.LinAlgError("Singular matrix")
        augmented[pivot_column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot_column],
        )

        pivot = augmented[pivot_column][pivot_column]
        augmented[pivot_column] = [value / pivot for value in augmented[pivot_column]]
        for row in range(size):
            if row == pivot_column:
                continue
            factor = augmented[row][pivot_column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[pivot_column]
                )
            ]

    return np.asarray([row[size:] for row in augmented], dtype=float)


def build_F(dt: Optional[float] = None) -> Matrix:
    """Build the constant-velocity state transition matrix F.

    Args:
        dt: Time step in seconds; default from tracking params.

    Returns:
        6x6 state transition matrix as a NumPy array.
    """
    if dt is None:
        dt = params.dt

    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def build_Q(dt: Optional[float] = None, q: Optional[float] = None) -> Matrix:
    """Build the process noise covariance matrix Q.

    Args:
        dt: Time step; default from tracking params.
        q: Process noise scale; default from tracking params.

    Returns:
        6x6 process noise matrix.
    """
    if dt is None:
        dt = params.dt
    if q is None:
        q = params.q

    q_diag = np.full(6, dt * q, dtype=float)
    return np.diag(q_diag)


def ekf_predict(
    x: Matrix,
    P: Matrix,
    F: Optional[Matrix] = None,
    Q: Optional[Matrix] = None,
) -> tuple[Matrix, Matrix]:
    """Predict state and covariance one time step forward.

    Args:
        x: State vector (6x1).
        P: State covariance (6x6).
        F: Optional transition matrix; build via ``build_F`` if None.
        Q: Optional process noise; build via ``build_Q`` if None.

    Returns:
        Tuple ``(x_pred, P_pred)``.
    """
    if F is None:
        F = build_F()
    if Q is None:
        Q = build_Q()

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    P = np.asarray(P, dtype=float)
    F = np.asarray(F, dtype=float)
    Q = np.asarray(Q, dtype=float)

    x_pred = _matmul(F, x)
    P_pred = _matmul(_matmul(F, P), F.T) + Q
    return x_pred, P_pred


def innovation(x: Matrix, meas: Any) -> Matrix:
    """Compute the measurement residual (innovation) gamma.

    Args:
        x: Predicted state.
        meas: Measurement with ``z`` and ``sensor.get_hx(x)``.

    Returns:
        Innovation vector ``z - h(x)``.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(meas.z, dtype=float).reshape(-1, 1)
    h_x = np.asarray(meas.sensor.get_hx(x), dtype=float).reshape(-1, 1)
    return z - h_x


def innovation_covariance(P: Matrix, meas: Any, H: Matrix) -> Matrix:
    """Compute the innovation covariance S = H P H' + R.

    Args:
        P: State covariance.
        meas: Measurement with ``R``.
        H: Measurement Jacobian.

    Returns:
        Innovation covariance matrix S.
    """
    P = np.asarray(P, dtype=float)
    H = np.asarray(H, dtype=float)
    R = np.asarray(meas.R, dtype=float)
    return _matmul(_matmul(H, P), H.T) + R


def ekf_update(x: Matrix, P: Matrix, meas: Any) -> tuple[Matrix, Matrix]:
    """Apply an EKF measurement update and return updated state and covariance.

    Args:
        x: Prior state.
        P: Prior covariance.
        meas: Associated measurement.

    Returns:
        Tuple ``(x_upd, P_upd)``.
    """
    x = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)
    H = np.asarray(meas.sensor.get_H(x), dtype=float)
    gamma = innovation(x, meas)
    S = innovation_covariance(P, meas, H)
    K = _matmul(_matmul(P, H.T), _inverse(S))

    I = np.eye(P.shape[0])
    residual_map = I - _matmul(K, H)
    x_upd = x + _matmul(K, gamma)
    P_upd = (
        _matmul(_matmul(residual_map, P), residual_map.T)
        + _matmul(_matmul(K, np.asarray(meas.R, dtype=float)), K.T)
    )
    return x_upd, P_upd
