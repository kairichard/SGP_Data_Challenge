from typing import Tuple

import numpy as np
from pandera.typing import DataFrame

from src.models.dataframes import PolarDataSchema


def get_boat_speed(
    polar_data: DataFrame[PolarDataSchema], wind_speed: float, wind_angle: float
) -> float:
    """
    Get boat speed from polar data for given wind speed and angle.

    Args:
        polar_data: DataFrame with columns TWA, TWS, BSP
        wind_speed: True wind speed
        wind_angle: True wind angle (absolute value)

    Returns:
        Expected boat speed in same units as polar data
    """
    # Ensure wind angle is positive and <= 180
    wind_angle = abs(wind_angle) % 180

    # Find nearest wind speed and angle in polar data
    tws_mask = (polar_data["TWS"] - wind_speed).abs().argsort()[:2]
    twa_mask = (polar_data["TWA"] - wind_angle).abs().argsort()[:2]

    # Get surrounding points for interpolation
    points = polar_data.iloc[list(set(tws_mask) & set(twa_mask))]

    if len(points) == 0:
        return 0.0

    # Simple average of nearest points
    return points["BSP"].mean()


def get_optimal_angles(
    polar_data: DataFrame[PolarDataSchema], wind_speed: float, mode: str
) -> Tuple[float, float]:
    """
    Get optimal sailing angles for upwind or downwind.

    Args:
        polar_data: DataFrame with columns TWA, TWS, BSP
        wind_speed: True wind speed
        mode: 'upwind' or 'downwind'

    Returns:
        Tuple of (port_twa, starboard_twa) optimal angles
    """
    # Filter to relevant wind speed
    tws_data = polar_data[(polar_data["TWS"] - wind_speed).abs() < 2.0]

    if mode == "upwind":
        # Find angle with best VMG between 30-60 degrees
        upwind_data = tws_data[(tws_data["TWA"] >= 30) & (tws_data["TWA"] <= 60)]
        best_twa = upwind_data.iloc[
            (upwind_data["BSP"] * np.cos(np.radians(upwind_data["TWA"]))).argmax()
        ]["TWA"]
        return -best_twa, best_twa

    else:  # downwind
        # Find angle with best VMG between 120-180 degrees
        downwind_data = tws_data[(tws_data["TWA"] >= 120) & (tws_data["TWA"] <= 180)]
        best_twa = downwind_data.iloc[
            (
                downwind_data["BSP"] * np.cos(np.radians(downwind_data["TWA"] - 180))
            ).argmax()
        ]["TWA"]
        return -best_twa, best_twa
