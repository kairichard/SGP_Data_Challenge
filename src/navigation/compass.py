import numpy as np
import pandas as pd


def calculate_compass_average(
    directions: np.ndarray, sample_rate: int, target_rate: int
) -> np.ndarray:
    """
    Calculate average compass directions at a lower sampling rate.

    Args:
        directions: Array of compass directions in degrees (0-360°)
        sample_rate: Current sampling rate in Hz (e.g., 1 for 1Hz)
        target_rate: Desired sampling rate in Hz (e.g., 0.1 for 10s average)

    Returns:
        Array of averaged compass directions at the target sampling rate
    """
    window_size = int(sample_rate / target_rate)
    averaged_directions = []

    for i in range(0, len(directions), window_size):
        window = directions[i : min(i + window_size, len(directions))]
        radians = np.deg2rad(window)
        x = np.cos(radians)
        y = np.sin(radians)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        mean_direction = np.rad2deg(np.arctan2(mean_y, mean_x))
        averaged_directions.append((mean_direction + 360) % 360)

    return pd.Series(averaged_directions)
