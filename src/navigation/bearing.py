from typing import Tuple

import numpy as np
import pyproj


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    geodesic = pyproj.Geod(ellps="WGS84")
    fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
    return fwd_azimuth


def calculate_vmc(
    boat_speed: float, boat_heading: float, course_bearing: float
) -> float:
    """Calculate VMC given boat speed and the difference between heading and course bearing"""
    angle_diff = np.abs(boat_heading - course_bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    return boat_speed * np.cos(np.radians(angle_diff))


def calculate_intersection(
    start_pos: Tuple[float, float],
    start_bearing: float,
    end_pos: Tuple[float, float],
    end_bearing: float,
) -> Tuple[float, float]:
    """
    Calculate intersection point of two lines given their start points and bearings.
    
    Visual explanation:
    
    Line 1:                    Line 2:
    start_pos ------>         end_pos ------>
    (lat1,lon1)               (lat2,lon2)
    bearing1                   bearing2
                   \ 
                    \
                     X <-- Returns this intersection point (lat3,lon3)
                    /
                   /
    
    Args:
        start_pos: (lat, lon) of first line start
        start_bearing: Bearing of first line in degrees
        end_pos: (lat, lon) of second line start
        end_bearing: Bearing of second line in degrees
        
    Returns:
        (lat, lon) of intersection point where the two lines cross
    """
    # Convert to radians
    lat1, lon1 = map(np.radians, start_pos)
    lat2, lon2 = map(np.radians, end_pos)
    brng1 = np.radians(start_bearing)
    brng2 = np.radians(end_bearing)

    # Calculate intersection point
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    dist12 = 2 * np.arcsin(
        np.sqrt(
            np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
    )
    if dist12 == 0:
        return start_pos

    # Initial/final bearings between points
    brngA = np.arccos(
        (np.sin(lat2) - np.sin(lat1) * np.cos(dist12)) / (np.sin(dist12) * np.cos(lat1))
    )
    brngB = np.arccos(
        (np.sin(lat1) - np.sin(lat2) * np.cos(dist12)) / (np.sin(dist12) * np.cos(lat2))
    )

    if np.sin(dlon) > 0:
        brng12 = brngA
        brng21 = 2 * np.pi - brngB
    else:
        brng12 = 2 * np.pi - brngA
        brng21 = brngB

    alpha1 = (brng1 - brng12 + np.pi) % (2 * np.pi) - np.pi
    alpha2 = (brng21 - brng2 + np.pi) % (2 * np.pi) - np.pi

    if np.sin(alpha1) == 0 and np.sin(alpha2) == 0:
        return start_pos  # Infinite intersections
    if np.sin(alpha1) * np.sin(alpha2) < 0:
        return start_pos  # Intersection is behind one or both points

    alpha3 = np.arccos(
        -np.cos(alpha1) * np.cos(alpha2)
        + np.sin(alpha1) * np.sin(alpha2) * np.cos(dist12)
    )
    dist13 = np.arctan2(
        np.sin(dist12) * np.sin(alpha1) * np.sin(alpha2),
        np.cos(alpha2) + np.cos(alpha1) * np.cos(alpha3),
    )
    lat3 = np.arcsin(
        np.sin(lat1) * np.cos(dist13) + np.cos(lat1) * np.sin(dist13) * np.cos(brng1)
    )
    dlon13 = np.arctan2(
        np.sin(brng1) * np.sin(dist13) * np.cos(lat1),
        np.cos(dist13) - np.sin(lat1) * np.sin(lat3),
    )
    lon3 = lon1 + dlon13

    return (np.degrees(lat3), np.degrees(lon3))
