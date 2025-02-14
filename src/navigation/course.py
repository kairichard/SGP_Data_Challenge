from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from src.models.dataframes import (BoatDataSchema, GPSPositionDataSchema,
                                   VMCAnnotationDataSchema)
from src.utils.gps import haversine_distance

from ..models.dataclasses import CompoundMark, Mark, Race
from .bearing import calculate_bearing, calculate_vmc


def find_next_mark(
    current_pos: Tuple[float, float], course: List[CompoundMark], current_mark_idx: int
) -> Tuple[int, Mark]:
    """
    Find the next mark to head towards based on current position
    Returns the mark index and the specific mark to aim for
    """
    if current_mark_idx >= len(course):
        return None, None

    next_compound = course[current_mark_idx]

    if len(next_compound.marks) > 1:
        mark1, mark2 = next_compound.marks
        bearing1 = calculate_bearing(
            current_pos[0], current_pos[1], mark1.lat, mark1.lon
        )
        bearing2 = calculate_bearing(
            current_pos[0], current_pos[1], mark2.lat, mark2.lon
        )

        return current_mark_idx, mark1 if bearing1 < bearing2 else mark2

    return current_mark_idx, next_compound.marks[0]


def is_mark_rounded(
    current_pos: Tuple[float, float],
    mark: CompoundMark,
    prev_pos: Optional[Tuple[float, float]] = None,
) -> bool:
    """
    Check if a mark has been rounded based on:
    1. Single mark: Within rounding zone
    2. Gate marks: Either crossed gate line or within rounding zone of either mark

    Args:
        current_pos: Current boat position (lat, lon)
        mark: CompoundMark to check
        prev_pos: Previous boat position for line crossing detection

    Returns:
        bool: True if mark has been rounded
    """
    ROUNDING_ZONE = 100.0  # Default rounding zone in meters
    MAX_POSITION_JUMP = 300.0  # Maximum allowed distance between positions in meters

    # Prevent large position jumps
    if prev_pos is not None:
        jump_distance = haversine_distance(current_pos, prev_pos)
        if jump_distance > MAX_POSITION_JUMP:
            return False

    if len(mark.marks) == 1:
        # Single mark - check if within rounding zone
        distance = haversine_distance(
            (current_pos[0], current_pos[1]), (mark.marks[0].lat, mark.marks[0].lon)
        )
        return distance <= ROUNDING_ZONE
    else:
        # Gate - check both line crossing and proximity
        if prev_pos is None:
            return False

        # Check if within rounding zone of either mark
        for gate_mark in mark.marks:
            distance = haversine_distance(
                (current_pos[0], current_pos[1]), (gate_mark.lat, gate_mark.lon)
            )
            if distance <= ROUNDING_ZONE:
                return True

        # Create gate line vectors
        gate_vector = np.array(
            [
                mark.marks[1].lat - mark.marks[0].lat,
                mark.marks[1].lon - mark.marks[0].lon,
            ]
        )

        # Create boat movement vector
        boat_vector = np.array(
            [current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1]]
        )

        # Check if vectors are not parallel (would indicate crossing)
        if np.abs(np.cross(gate_vector, boat_vector)) > 1e-10:
            # Create vectors from gate start to boat positions
            prev_vector = np.array(
                [prev_pos[0] - mark.marks[0].lat, prev_pos[1] - mark.marks[0].lon]
            )

            current_vector = np.array(
                [current_pos[0] - mark.marks[0].lat, current_pos[1] - mark.marks[0].lon]
            )

            # Check if boat crossed the gate line
            prev_side = np.cross(gate_vector, prev_vector)
            current_side = np.cross(gate_vector, current_vector)

            return np.sign(prev_side) != np.sign(current_side)

        return False


@pa.check_types(lazy=True)
def calculate_track_vmc(
    boat_data: DataFrame[GPSPositionDataSchema], course: List[CompoundMark]
) -> DataFrame[VMCAnnotationDataSchema]:
    """
    Calculate VMC (Velocity Made good on Course) for each point in the boat's track
    using pre-calculated mark tracking data.

    Args:
        boat_data: DataFrame with boat position and mark tracking columns
        course: List of course marks (used for finish line detection)

    Returns:
        DataFrame with VMC and mark information
    """
    results = []

    for idx, row in boat_data.iterrows():
        # Get current position and mark info from pre-calculated columns
        current_pos = (row["LATITUDE_GPS"], row["LONGITUDE_GPS"])

        # Check if boat has finished
        if row["CURRENT_LEG_NUM"] == 7:
            results.append(
                {
                    "VMC_km_h_1": 0.0,
                }
            )
            continue

        # Calculate bearing to next mark
        next_mark = course[row["CURRENT_LEG_NUM"]].marks[0]
        course_bearing = calculate_bearing(
            current_pos[0], current_pos[1], next_mark.lat, next_mark.lon
        )

        # Calculate VMC using boat speed and heading
        vmc = calculate_vmc(
            boat_speed=row["SPEED"],
            boat_heading=row["HEADING"],
            course_bearing=course_bearing,
        )

        results.append(
            {
                "VMC_km_h_1": vmc,
            }
        )

    return pd.DataFrame(results, index=boat_data.index)


def add_own_mark_tracking(
    boat_data: DataFrame[BoatDataSchema], course: List[CompoundMark]
) -> DataFrame[BoatDataSchema]:
    """
    Add mark tracking columns to boat data:
    - next_mark: Name of the next mark to round
    - mark_distance: Distance to next mark in meters
    - current_mark_idx: Index of current mark

    Args:
        boat_data: DataFrame with boat position data
        course: List of course marks

    Returns:
        DataFrame with added mark tracking columns
    """
    results = []
    current_mark_idx = 0
    prev_pos = None

    boat_data["DATETIME"] = pd.to_datetime(boat_data["DATETIME"], utc=True)
    for idx, row in boat_data.iterrows():
        current_pos = (row["LATITUDE_GPS_unk"], row["LONGITUDE_GPS_unk"])

        # Check if mark has been rounded
        if prev_pos and current_mark_idx < len(course) - 1:
            if is_mark_rounded(current_pos, course[current_mark_idx], prev_pos):
                current_mark_idx += 1

        # Get next mark info
        if current_mark_idx >= len(course):
            results.append(
                {
                    "NEXT_MARK": "FINISHED",
                    "MARK_DISTANCE": 0.0,
                    "CURRENT_LEG_NUM": current_mark_idx,
                }
            )
        else:
            next_mark = course[current_mark_idx].marks[0]
            mark_distance = haversine_distance(
                current_pos, (next_mark.lat, next_mark.lon)
            )

            results.append(
                {
                    "NEXT_MARK": next_mark.name,
                    "MARK_DISTANCE": mark_distance,
                    "CURRENT_LEG_NUM": current_mark_idx,
                }
            )

        prev_pos = current_pos

    # Add results as new columns
    result_df = pd.DataFrame(results, index=boat_data.index)
    return pd.concat([boat_data, result_df], axis=1)


# @pa.check_types(lazy=True)
def trim_to_race(
    boat_data: DataFrame[BoatDataSchema], race_course: Race
) -> DataFrame[BoatDataSchema]:
    """
    Filter boat data to only include data from the specified race.

    Args:
        boat_data: DataFrame with boat position and mark tracking data
        race_course: RaceCourse object containing race metadata and marks

    Returns:
        DataFrame filtered to only include data from the specified race

    Note:
        Requires boat_data to have 'next_mark' column from add_mark_tracking()
        and 'TRK_RACE_NUM_unk' column identifying the race number
    """
    # Filter to only include data from this race
    race_data = boat_data[
        boat_data["TRK_RACE_NUM_unk"] == race_course.race_id + ".0"
    ].copy()
    return race_data.copy()


# @pa.check_types(lazy=True)
def trim_to_start_finish(
    boat_data: DataFrame[BoatDataSchema], race_course: Race
) -> DataFrame[BoatDataSchema]:
    """
    Filter boat data to only include data between race start and finish.

    Args:
        boat_data: DataFrame with boat position and race tracking data

    Returns:
        DataFrame filtered to only include data between start and finish transitions
    """
    boat_data = trim_to_race(boat_data, race_course)
    # Find first transition from 0 to 1 (race start)

    start_mask = (boat_data["TRK_LEG_NUM_unk"].shift(1) == 0) & (
        boat_data["TRK_LEG_NUM_unk"] == 1
    )
    if not start_mask.any():
        return boat_data.copy()
    start_idx = start_mask.idxmax()

    # Find first transition from 7 to 0 (race finish) after start
    finish_mask = (boat_data["TRK_LEG_NUM_unk"].shift(1) == 7) & (
        boat_data["TRK_LEG_NUM_unk"] == 0
    )
    finish_mask = finish_mask[start_idx:]
    if not finish_mask.any():
        return boat_data.loc[start_idx:].copy()
    finish_idx = finish_mask.idxmax()

    # Return data between start and finish
    return boat_data.loc[start_idx:finish_idx].copy()
