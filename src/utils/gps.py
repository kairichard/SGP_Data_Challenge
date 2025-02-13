from typing import List, Tuple

import numpy as np


def filter_gps_jumps(
    points: List[Tuple[float, float]], max_jump_meters: float = 100.0
) -> List[Tuple[float, float]]:
    """
    Filter out large GPS jumps from a series of points
    Args:
        points: List of (lat, lon) tuples
        max_jump_meters: Maximum allowed distance between consecutive points
    Returns:
        Filtered list of points
    """
    if len(points) < 2:
        return points

    filtered_points = [points[0]]  # Keep first point

    for i in range(1, len(points)):
        distance = haversine_distance(
            (filtered_points[-1][0], filtered_points[-1][1]),
            (points[i][0], points[i][1]),
        )

        if distance <= max_jump_meters:
            filtered_points.append(points[i])
    return filtered_points


def haversine_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """Calculate distance between two lat/lon points in meters"""
    R = 6371000  # Earth's radius in meters
    lat1, lon1 = map(np.radians, point1)
    lat2, lon2 = map(np.radians, point2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c
