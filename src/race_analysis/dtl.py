"""
Distance to Leader (DTL) calculation module.

This module provides functionality to calculate the Distance to Leader (DTL) metric
for each boat in a race. The DTL metric represents how far behind the leader a boat
is at any given time, taking into account:

Key Functions:
- calculate_dtl(): Main function to compute DTL for a boat
- calculate_dtl_for_fleet(): Calculate DTL for all boats in the fleet

The module uses:
- Existing haversine_distance from utils.gps
- Course navigation from navigation.course
- Progress tracking from race_analysis.progress

DTL is calculated by:
1. Identifying the current leader
2. Computing direct distance to leader
3. Adjusting for course layout and different legs
4. Accounting for tactical considerations
"""

from typing import List

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from src.models.dataframes import (DTLDataSchema, GPSPositionDataSchema,
                                   LeaderDataSchema)

from ..models.dataclasses import CompoundMark
from ..utils.gps import haversine_distance


@pa.check_types
def calculate_dtl(
    position_data: DataFrame[GPSPositionDataSchema],
    leader_data: DataFrame[LeaderDataSchema],
    course: List[CompoundMark],
) -> DataFrame[DTLDataSchema]:
    """
    Calculate Distance to Leader (DTL) for a boat.

    Args:
        boat_data: DataFrame with boat position and data
        leader_data: DataFrame with leader information from identify_leader()
        course: List of course marks defining the race course

    Returns:
        DataFrame with DTL metrics:
            - timestamp: datetime
            - dtl_direct: float (direct distance to leader in meters)
            - dtl_course: float (distance along course to leader in meters)
            - leg_behind: int (number of legs behind leader)
            - progress_diff: float (difference in leg progress, -1 to 1)
    """
    results = []

    # Ensure timestamps are datetime
    for idx, row in position_data.iterrows():
        # Find corresponding leader data
        leader_idx = leader_data["DATETIME"].searchsorted(row["DATETIME"])
        if leader_idx >= len(leader_data):
            continue

        leader_row = leader_data.iloc[leader_idx]

        # Get boat position and progress
        boat_pos = (row["LATITUDE_GPS"], row["LONGITUDE_GPS"])
        leader_pos = (leader_row["LEADER_LAT"], leader_row["LEADER_LON"])

        # Calculate direct distance to leader
        dtl_direct = haversine_distance(boat_pos, leader_pos)

        # Calculate boat's progress
        mark_idx = row["CURRENT_LEG_NUM"]
        next_mark = course[mark_idx].marks[0]

        # Calculate progress on current leg
        dist_to_next = haversine_distance(boat_pos, (next_mark.lat, next_mark.lon))

        # Skip indirect DTL calculation if this is the leader
        # Calculate leader's distance to their next mark
        leader_dist_to_next = haversine_distance(
            leader_pos, (next_mark.lat, next_mark.lon)
        )

        # Calculate indirect DTL as sum of distances to respective next marks
        dtl_indirect = dist_to_next + leader_dist_to_next

        # Calculate leg difference and progress
        leg_behind = leader_row["LEADER_LEG"] - mark_idx

        # If on same leg, use direct distance
        if mark_idx == leader_row["LEADER_LEG"]:
            dtl_indirect = dtl_direct

        if row["DATETIME"] == leader_row["DATETIME"] and dtl_direct < 1:
            dtl_indirect = 0
            leg_behind = 0

        results.append(
            {
                "DATETIME": row["DATETIME"],
                "DTL_DIRECT": dtl_direct,
                "DTL_INDIRECT": dtl_indirect,
                "LEG_BEHIND": leg_behind,
            }
        )

    return pd.DataFrame(results)
