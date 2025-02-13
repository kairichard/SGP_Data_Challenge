from src.models.dataclasses import CompoundMark
from src.navigation.bearing import calculate_bearing


def determine_leg_type(
    start_mark: CompoundMark, end_mark: CompoundMark, wind_direction: float
) -> str:
    """
    Determine leg type based on mark sequence and wind direction.

    Args:
        start_mark: Starting mark
        end_mark: End mark
        wind_direction: True wind direction in degrees

    Returns:
        str: 'upwind', 'downwind', or 'reaching'
    """
    # Calculate basic leg bearing
    leg_bearing = calculate_bearing(
        start_mark.marks[0].lat,
        start_mark.marks[0].lon,
        end_mark.marks[0].lat,
        end_mark.marks[0].lon,
    )

    # Calculate true wind angle relative to leg bearing
    relative_wind = (wind_direction - leg_bearing) % 360

    # First determine by mark sequence
    if start_mark.name == "SL1" and end_mark.name == "M1":
        # Start to first mark is always upwind
        return "upwind"

    elif start_mark.name == "M1" and end_mark.name.startswith("LG"):
        # M1 to first leeward gate is downwind
        return "downwind"

    elif start_mark.name.startswith("LG") and end_mark.name.startswith("WG"):
        # Leeward to windward gate is upwind
        return "upwind"

    elif start_mark.name.startswith("WG") and end_mark.name.startswith("LG"):
        # Windward to leeward gate is downwind
        return "downwind"

    # For legs not covered by mark sequence rules, determine type based on wind angle
    # Note: TWA is measured from wind direction to boat heading
    # - TWA 0-90° or 270-360° = Upwind
    # - TWA 30-150° or 210-330° = Reaching
    # - TWA 150-210° = Downwind
    #
    if (0 <= relative_wind <= 90) or (270 <= relative_wind <= 360):
        # Wind is coming from ahead - boat must tack upwind
        # Example: Wind at 0°, boat at 45° = close hauled sailing
        return "upwind"
    elif 150 <= relative_wind <= 210:
        # Wind is coming from behind - boat must gybe downwind
        # Example: Wind at 180°, boat at 160° = running
        return "downwind"
    else:
        # Wind is more from the side - reaching conditions
        # Example: Wind at 90°, boat at 120° = beam reach
        return "reaching"
