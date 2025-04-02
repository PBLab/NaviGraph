"""
General conversion utilities for working with angles and quaternions.
These functions are independent and can be reused by any module.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from typing import Any

def wrap_angle(angle: float) -> float:
    """
    Wrap an angle in degrees to the range [-180, 180].

    Args:
        angle (float): Angle in degrees.

    Returns:
        float: Angle wrapped to [-180, 180].
    """
    return (angle + 180) % 360 - 180

def quaternions_to_euler(data: pd.DataFrame, yaw_offset: float = -167, positive_direction: float = -1) -> np.ndarray:
    """
    Convert quaternion values in a DataFrame to Euler angles (yaw, pitch, roll).

    The DataFrame is expected to contain columns: 'qw', 'qx', 'qy', 'qz'. The conversion uses the ZYX
    convention, outputs angles in degrees, applies a yaw offset, and multiplies yaw by positive_direction.

    Args:
        data (pd.DataFrame): DataFrame with quaternion columns.
        yaw_offset (float): Offset to subtract from yaw (in degrees).
        positive_direction (float): Multiplier for yaw to adjust its sign.

    Returns:
        np.ndarray: An array of shape (N, 3) with Euler angles.
    """
    # Drop rows with missing quaternion data.
    quats = data[['qw', 'qx', 'qy', 'qz']].dropna().values
    # Reorder quaternions to [qx, qy, qz, qw] for scipy's Rotation.
    quats_reordered = quats[:, [1, 2, 3, 0]]
    # Convert to Euler angles (ZYX: yaw, pitch, roll) in degrees.
    euler_angles = Rotation.from_quat(quats_reordered).as_euler('zyx', degrees=True)
    # Adjust yaw: apply offset, then wrap, then multiply.
    euler_angles[:, 0] = wrap_angle(euler_angles[:, 0] - yaw_offset)
    euler_angles[:, 0] *= positive_direction
    return euler_angles
