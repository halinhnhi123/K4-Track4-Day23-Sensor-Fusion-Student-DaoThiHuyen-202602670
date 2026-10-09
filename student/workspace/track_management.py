"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

params = get_tracking_params()


def _matmul(left: Any, right: Any) -> np.ndarray:
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


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score``.
    """
    rotation = np.asarray(meas.sensor.sens_to_veh, dtype=float)[:3, :3]
    translation = np.asarray(meas.sensor.sens_to_veh, dtype=float)[:3, 3]
    position = _matmul(rotation, np.asarray(meas.z, dtype=float).reshape(3, 1))
    position += translation.reshape(3, 1)

    covariance = np.zeros((params.dim_state, params.dim_state), dtype=float)
    covariance[:3, :3] = _matmul(
        _matmul(rotation, np.asarray(meas.R, dtype=float)), rotation.T
    )
    velocity_variances = np.square(
        [params.sigma_p44, params.sigma_p55, params.sigma_p66]
    )
    covariance[3:, 3:] = np.diag(velocity_variances)

    state = np.vstack((position, np.zeros((params.dim_state - 3, 1))))
    return {
        "x": np.asmatrix(state),
        "P": np.asmatrix(covariance),
        "state": "initialized",
        "score": 1.0 / params.window,
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    increment = 1.0 / params.window
    score = float(track["score"])
    if associated:
        score = min(1.0, score + increment)
    else:
        score = max(0.0, score - increment)

    state = track["state"]
    if state != "confirmed":
        if score > params.confirmed_threshold:
            state = "confirmed"
        elif associated:
            state = "tentative"

    track["score"] = score
    track["state"] = state
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    covariance = np.asarray(track["P"], dtype=float)
    if covariance[0, 0] > params.max_P or covariance[1, 1] > params.max_P:
        return True
    if track["state"] == "confirmed":
        return bool(track["score"] < params.delete_threshold)
    return bool(track["score"] <= 0)
