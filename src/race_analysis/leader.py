"""
Race Leader Identification Module.

This module handles identifying the current race leader at any given timestamp,
considering boats' positions, progress, and course layout.

Key Functions:
- identify_leader(): Determine race leader at each timestamp
- validate_leader_changes(): Verify leader transitions
- calculate_leader_stats(): Generate leader statistics

The module considers:
1. Boat positions relative to course
2. Progress on current leg
3. Mark roundings
4. Course boundaries
5. Different legs/tactics

Uses existing functionality from:
- navigation.course for course positioning
- race_analysis.progress for progress tracking
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from src.models.dataclasses import CompoundMark
from src.models.dataframes import GPSPositionDataSchema, LeaderDataSchema

from ..utils.gps import haversine_distance


@pa.check_types
def identify_leader(
    all_boat_data: Dict[str, DataFrame[GPSPositionDataSchema]],
    course: List[CompoundMark],
) -> DataFrame[LeaderDataSchema]:
    """
    Determine race leader at each timestamp, considering:
    - Race only starts when boats cross start line
    - Leader is boat closest to finish after first boat finishes
    - Race ends when all boats finish
    - Even stop processing after all boats have finished
    - Leader is the boat with the lowest distance to the next mark and the highest mark number
    """
    # Process and align timestamps
    timestamps = get_aligned_timestamps(all_boat_data)
    leader_data = []

    # Track boat progress

    for timestamp in timestamps:
        boat_positions = get_positions_at_timestamp(timestamp, all_boat_data, course)

        if not boat_positions:  # No valid positions at this timestamp
            continue

        leader_id = determine_leader(boat_positions)
        if leader_id:
            leader = boat_positions[leader_id]
            leader_data.append(create_leader_entry(timestamp, leader_id, leader))

    return pd.DataFrame(leader_data)


@pa.check_types
def get_aligned_timestamps(
    all_boat_data: Dict[str, DataFrame[GPSPositionDataSchema]]
) -> List[datetime]:
    """Get sorted list of all unique timestamps from boat data."""
    timestamps = set()
    for boat_data in all_boat_data.values():
        timestamps.update(boat_data["DATETIME"])
    return sorted(list(timestamps))


@pa.check_types
def get_positions_at_timestamp(
    timestamp: datetime,
    all_boat_data: Dict[str, DataFrame[GPSPositionDataSchema]],
    course: List[CompoundMark],
) -> Dict[str, Dict]:
    """Get all valid boat positions and their progress at given timestamp."""
    boat_positions = {}

    for boat_id, boat_data in all_boat_data.items():

        data = position_data_at_timestamp(boat_data, timestamp)
        if data is None:
            continue

        next_mark = course[data["CURRENT_LEG_NUM"]].marks[0]
        pos = (data["LATITUDE_GPS"], data["LONGITUDE_GPS"])
        boat_positions[boat_id] = {
            "position": pos,
            "mark_idx": data["CURRENT_LEG_NUM"],
            "dist_to_mark": haversine_distance(pos, (next_mark.lat, next_mark.lon)),
        }

    return boat_positions


def position_data_at_timestamp(
    boat_data: DataFrame[GPSPositionDataSchema], timestamp: datetime
) -> Optional[Tuple[float, float]]:
    """Extract boat position at specific timestamp."""
    mask = boat_data["DATETIME"] == timestamp
    if not mask.any():
        return None
    row = boat_data[mask].iloc[0]
    return row


def determine_leader(boat_positions: Dict[str, Dict]) -> Optional[str]:
    """Determine leader based on mark index and distance to next mark."""
    if not boat_positions:
        return None

    return min(
        boat_positions.keys(),
        key=lambda bid: (
            -boat_positions[bid]["mark_idx"],  # Higher mark index is better
            boat_positions[bid]["dist_to_mark"],  # Lower distance is better
        ),
    )


def create_leader_entry(timestamp: datetime, leader_id: str, leader_info: Dict) -> Dict:
    """Create a standardized leader data entry."""
    return {
        "DATETIME": timestamp,
        "LEADER_ID": leader_id,
        "LEADER_LAT": leader_info["position"][0],
        "LEADER_LON": leader_info["position"][1],
        "LEADER_LEG": leader_info["mark_idx"],
    }
