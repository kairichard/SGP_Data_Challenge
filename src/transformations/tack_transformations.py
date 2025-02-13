
import pandera as pa
from pandera.typing import DataFrame

from src.models.dataframes import BoatDataSchema, BoatDataTackedSchema


@pa.check_types
def normalize_to_starboard_tack(
    df: DataFrame[BoatDataSchema],
) -> DataFrame[BoatDataTackedSchema]:
    """
    Transform F50 telemetry data to be normalized to starboard tack perspective.

    Args:
        df: DataFrame with original F50 telemetry data

    Returns:
        DataFrame with additional tack-normalized columns
    """
    # Create copy to avoid modifying original
    result_df = df.copy()

    # Determine tack (True for port tack)
    is_port = df["TWA_SGP_deg"] < 0

    # Apply transformations based on tack

    # 1. Angles that need to be negated on port tack
    negate_columns = {
        "TWA_SGP_deg": "TWA_TACKED_deg",
        "AWA_SGP_deg": "AWA_TACKED_deg",
        "RATE_YAW_deg_s_1": "RATE_YAW_TACKED_deg_s_1",
        "LEEWAY_deg": "LEEWAY_TACKED_deg",
        "ANGLE_CA1_deg": "ANGLE_CA1_TACKED_deg",
        "ANGLE_CA2_deg": "ANGLE_CA2_TACKED_deg",
        "ANGLE_CA3_deg": "ANGLE_CA3_TACKED_deg",
        "ANGLE_CA4_deg": "ANGLE_CA4_TACKED_deg",
        "ANGLE_CA5_deg": "ANGLE_CA5_TACKED_deg",
        "ANGLE_CA6_deg": "ANGLE_CA6_TACKED_deg",
        "ANGLE_WING_TWIST_deg": "ANGLE_WING_TWIST_TACKED_deg",
        "ANGLE_WING_ROT_deg": "ANGLE_WING_ROT_TACKED_deg",
        "HEEL_deg": "HEEL_TACKED_deg",
        "ANGLE_RUDDER_deg": "ANGLE_RUDDER_TACKED_deg",
        "ANGLE_RUD_AVG_deg": "ANGLE_RUD_AVG_TACKED_deg",
        "ANGLE_RUD_DIFF_TACK_deg": "ANGLE_RUD_DIFF_TACK_TACKED_deg",
    }

    for orig_col, new_col in negate_columns.items():
        result_df[new_col] = df[orig_col].where(~is_port, -df[orig_col])

    # 2. Angles that need 180° rotation on port tack
    rotate_columns = {
        "HEADING_deg": "HEADING_TACKED_deg",
        "GPS_COG_deg": "GPS_COG_TACKED_deg",
    }

    for orig_col, new_col in rotate_columns.items():
        result_df[new_col] = df[orig_col].where(~is_port, (df[orig_col] + 180) % 360)

    # 3. Port/Starboard measurements that need to be swapped on port tack
    # Ride heights
    result_df["LENGTH_RH_WINDWARD_mm"] = df.apply(
        lambda row: (
            row["LENGTH_RH_S_mm"] if row["TWA_SGP_deg"] < 0 else row["LENGTH_RH_P_mm"]
        ),
        axis=1,
    )
    result_df["LENGTH_RH_LEEWARD_mm"] = df.apply(
        lambda row: (
            row["LENGTH_RH_P_mm"] if row["TWA_SGP_deg"] < 0 else row["LENGTH_RH_S_mm"]
        ),
        axis=1,
    )

    # Daggerboard settings
    result_df["ANGLE_DB_RAKE_WINDWARD_deg"] = df.apply(
        lambda row: (
            row["ANGLE_DB_RAKE_S_deg"]
            if row["TWA_SGP_deg"] < 0
            else row["ANGLE_DB_RAKE_P_deg"]
        ),
        axis=1,
    )
    result_df["ANGLE_DB_RAKE_LEEWARD_deg"] = df.apply(
        lambda row: (
            row["ANGLE_DB_RAKE_P_deg"]
            if row["TWA_SGP_deg"] < 0
            else row["ANGLE_DB_RAKE_S_deg"]
        ),
        axis=1,
    )

    result_df["ANGLE_DB_CANT_WINDWARD_deg"] = df.apply(
        lambda row: (
            row["ANGLE_DB_CANT_S_deg"]
            if row["TWA_SGP_deg"] < 0
            else row["ANGLE_DB_CANT_P_deg"]
        ),
        axis=1,
    )
    result_df["ANGLE_DB_CANT_LEEWARD_deg"] = df.apply(
        lambda row: (
            row["ANGLE_DB_CANT_P_deg"]
            if row["TWA_SGP_deg"] < 0
            else row["ANGLE_DB_CANT_S_deg"]
        ),
        axis=1,
    )

    return result_df
