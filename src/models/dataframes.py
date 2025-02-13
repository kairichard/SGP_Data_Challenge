from typing import Annotated

import pandas as pd
import pandera as pa
from pandera.typing import Series


class CompassDataSchema(pa.DataFrameModel):
    DATETIME: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    DIRECTION: Series[float]


class GPSPositionDataSchema(pa.DataFrameModel):
    DATETIME: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    LATITUDE_GPS: Series[float]
    LONGITUDE_GPS: Series[float]
    CURRENT_LEG_NUM: Series[int]
    SPEED: Series[float]
    HEADING: Series[float]


class VMCAnnotationDataSchema(pa.DataFrameModel):
    VMC_km_h_1: Series[float]


class LeaderDataSchema(pa.DataFrameModel):
    DATETIME: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    LEADER_ID: Series[str]
    LEADER_LAT: Series[float]
    LEADER_LON: Series[float]
    LEADER_LEG: Series[int]


class GPSPositionDataSchemaWithBoatID(GPSPositionDataSchema):
    BOAT_ID: Series[str]


class DTLDataSchema(pa.DataFrameModel):
    DATETIME: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    DTL_DIRECT: Series[float]
    DTL_INDIRECT: Series[float]
    LEG_BEHIND: Series[int]


class VMCVisualizationDataSchema(GPSPositionDataSchema, VMCAnnotationDataSchema):
    pass


class DTLVisualizationDataSchema(GPSPositionDataSchema, DTLDataSchema):
    pass


class PolarDataSchema(pa.DataFrameModel):
    """Schema for polar performance data"""

    TWA: Series[float]  # True wind angle
    TWS: Series[float]  # True wind speed
    BSP: Series[float]  # Boat speed


class BoatDataSchema(pa.DataFrameModel):
    """Schema for F50 telemetry data with tack-normalized measurements"""

    # Identifiers and timestamps
    BOAT: Series[str]
    DATETIME: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    ## not correct timezone
    TIME_LOCAL_unk: Series[pa.dtypes.Timestamp]

    # GPS and position
    LATITUDE_GPS_unk: Series[float]
    LONGITUDE_GPS_unk: Series[float]

    # Speed and motion
    BOAT_SPEED_km_h_1: Series[float]
    GPS_SOG_km_h_1: Series[float]
    VMG_km_h_1: Series[float]

    # Angles and directions
    HEADING_deg: Series[float]  # 0-360
    TWA_SGP_deg: Series[float]  # -180 to +180
    TWS_SGP_km_h_1: Series[float]
    TWD_SGP_deg: Series[float]  # 0-360
    AWA_SGP_deg: Series[float]  # -180 to +180
    RATE_YAW_deg_s_1: Series[float]
    GPS_COG_deg: Series[float]  # 0-360
    LEEWAY_deg: Series[float]

    # Control surfaces
    ANGLE_CA1_deg: Series[float]
    ANGLE_CA2_deg: Series[float]
    ANGLE_CA3_deg: Series[float]
    ANGLE_CA4_deg: Series[float]
    ANGLE_CA5_deg: Series[float]
    ANGLE_CA6_deg: Series[float]

    # Wing controls
    ANGLE_WING_TWIST_deg: Series[float]
    ANGLE_WING_ROT_deg: Series[float]

    # Boat attitude
    PITCH_deg: Series[float]
    HEEL_deg: Series[float]

    # Ride heights
    LENGTH_RH_P_mm: Series[float]
    LENGTH_RH_S_mm: Series[float]
    LENGTH_RH_BOW_mm: Series[float]

    # Rudder and daggerboard controls
    ANGLE_RUDDER_deg: Series[float]
    ANGLE_RUD_AVG_deg: Series[float]
    ANGLE_RUD_DIFF_TACK_deg: Series[float]
    ANGLE_DB_RAKE_P_deg: Series[float]
    ANGLE_DB_RAKE_S_deg: Series[float]
    ANGLE_DB_CANT_P_deg: Series[float]
    ANGLE_DB_CANT_S_deg: Series[float]

    # Race tracking
    TRK_LEG_NUM_TOT_unk: Series[int]
    TRK_LEG_NUM_unk: Series[int]
    TRK_RACE_NUM_unk: Series[str]

    # Performance calculations
    PC_TTS_s: Series[float]
    PC_TTK_s: Series[float]
    PC_TTB_s: Series[float]
    PC_DTB_m: Series[float]
    PC_DTL_m: Series[float]
    PC_TTM_s: Series[float]


class BoatDataTackedSchema(BoatDataSchema):
    """Schema for F50 telemetry data with additional tack-normalized measurements.
    Columns with '_TACKED' suffix represent measurements normalized to starboard tack.
    """

    # Angles and directions (tack-normalized)
    HEADING_TACKED_deg: Series[float]  # rotated 180° on port tack
    TWA_TACKED_deg: Series[float]  # negated on port tack
    AWA_TACKED_deg: Series[float]  # negated on port tack
    RATE_YAW_TACKED_deg_s_1: Series[float]  # negated on port tack
    GPS_COG_TACKED_deg: Series[float]  # rotated 180° on port tack
    LEEWAY_TACKED_deg: Series[float]  # negated on port tack

    # Camber angles (negated on port tack)
    ANGLE_CA1_TACKED_deg: Series[float]  # Camber angle 1 - normalized to starboard tack
    ANGLE_CA2_TACKED_deg: Series[float]  # Camber angle 2 - normalized to starboard tack
    ANGLE_CA3_TACKED_deg: Series[float]  # Camber angle 3 - normalized to starboard tack
    ANGLE_CA4_TACKED_deg: Series[float]  # Camber angle 4 - normalized to starboard tack
    ANGLE_CA5_TACKED_deg: Series[float]  # Camber angle 5 - normalized to starboard tack
    ANGLE_CA6_TACKED_deg: Series[float]  # Camber angle 6 - normalized to starboard tack

    # Wing controls (negated on port tack)
    ANGLE_WING_TWIST_TACKED_deg: Series[float]
    ANGLE_WING_ROT_TACKED_deg: Series[float]

    # Boat attitude
    HEEL_TACKED_deg: Series[float]  # negated on port tack

    # Ride heights (swapped on port tack)
    LENGTH_RH_WINDWARD_mm: Series[float]  # always windward side
    LENGTH_RH_LEEWARD_mm: Series[float]  # always leeward side

    # Rudder and daggerboard controls
    ANGLE_RUDDER_TACKED_deg: Series[float]  # negated on port tack
    ANGLE_RUD_AVG_TACKED_deg: Series[float]  # negated on port tack
    ANGLE_RUD_DIFF_TACK_TACKED_deg: Series[float]  # negated on port tack

    # Daggerboard controls (swapped on port tack)
    ANGLE_DB_RAKE_WINDWARD_deg: Series[float]  # always windward side
    ANGLE_DB_RAKE_LEEWARD_deg: Series[float]  # always leeward side
    ANGLE_DB_CANT_WINDWARD_deg: Series[float]  # always windward side
    ANGLE_DB_CANT_LEEWARD_deg: Series[float]  # always leeward side


class ManeuverDataSchema(pa.DataFrameModel):
    """Schema for F50 maneuver analysis data"""

    # Identifiers and timestamps
    BOAT: Series[str]
    HULL: Series[str]
    WING_CONFIG_unk: Series[float]
    MD4_SEL_DB_unk: Series[float]
    MD4_SEL_RUD_unk: Series[float]
    DATETIME: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    TIME_LOCAL_unk: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]

    # Race context
    race: Series[str]
    leg: Series[int]
    type: Series[str]  # tack or gybe

    # Boat speed metrics
    entry_bsp: Series[float]  # km/h
    exit_bsp: Series[float]  # km/h
    min_bsp: Series[float]  # km/h
    bsp_loss: Series[float]  # km/h

    # Wind angles
    entry_twa: Series[float]  # degrees
    exit_twa: Series[float]  # degrees
    orig_entry_twa: Series[float]  # degrees
    orig_exit_twa: Series[float]  # degrees

    # Ride heights
    entry_rh: Series[float] = pa.Field(nullable=True, coerce=True)
    exit_rh: Series[float] = pa.Field(nullable=True, coerce=True)
    entry_rh_stability: Series[float] = pa.Field(nullable=True, coerce=True)
    turn_min_rh: Series[float] = pa.Field(nullable=True, coerce=True)

    # Maneuver execution
    max_yaw_rate: Series[float]  # degrees/second
    turning_time: Series[float]  # seconds
    t_swap: Series[float]  # seconds
    max_rudder_angle: Series[float]  # degrees

    # Flight state
    flying: Series[int]  # binary 0/1
    pop_time: Series[float] = pa.Field(nullable=True, coerce=True)

    # Environmental conditions
    tws: Series[float]  # knots
    avg_TWD: Series[float]  # degrees

    # G-forces
    max_lat_gforce: Series[float] = pa.Field(nullable=True, coerce=True)
    max_fwd_gforce: Series[float] = pa.Field(nullable=True, coerce=True)
    max_gforce: Series[float] = pa.Field(nullable=True, coerce=True)

    # Board timing
    db_down: Series[float]  # seconds
    two_DB_time: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds
    two_DB_Broadcast: Series[float]  # seconds
    drop_time_P: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds
    drop_time_S: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds
    unstow_time_P: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds
    unstow_time_S: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds
    stow_time_P: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds
    stow_time_S: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds
    boards_up_time_S: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds
    boards_up_time_P: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds

    # Pressure systems
    press_sys_acc_start: Series[float]  # PSI
    press_sys_acc_end: Series[float]  # PSI
    press_sys_acc_delta: Series[float]  # PSI
    press_rake_acc_start: Series[float]  # PSI
    press_rake_acc_end: Series[float]  # PSI
    press_rake_acc_delta: Series[float]  # PSI
    pump_press_avg: Series[float]  # PSI
    pump_press_max: Series[float]  # PSI
    press_wing_acc_start: Series[float]  # PSI
    press_wing_acc_end: Series[float]  # PSI

    # Daggerboard pressures
    db_ud_ret_press_s_avg: Series[float]  # PSI
    db_ud_ret_press_s_max: Series[float]  # PSI
    db_ud_ret_press_p_avg: Series[float]  # PSI
    db_ud_ret_press_p_max: Series[float]  # PSI
    db_ud_ext_press_s_avg: Series[float]  # PSI
    db_ud_ext_press_s_max: Series[float]  # PSI
    db_ud_ext_press_p_avg: Series[float]  # PSI
    db_ud_ext_press_p_max: Series[float]  # PSI
    db_cant_ret_press_p_avg: Series[float]  # PSI
    db_cant_ret_press_p_max: Series[float]  # PSI
    db_cant_ret_press_s_avg: Series[float]  # PSI
    db_cant_ret_press_s_max: Series[float]  # PSI
    db_cant_ext_press_p_avg: Series[float]  # PSI
    db_cant_ext_press_p_max: Series[float]  # PSI
    db_cant_ext_press_s_avg: Series[float]  # PSI
    db_cant_ext_press_s_max: Series[float]  # PSI

    # Boat attitude
    entry_heel: Series[float]  # degrees
    entry_pitch: Series[float]  # degrees
    exit_heel: Series[float]  # degrees
    exit_pitch: Series[float]  # degrees
    heel_at_drop: Series[float]  # degrees
    pitch_at_drop: Series[float]  # degrees

    # Jib controls
    entry_jib_lead: Series[float]
    exit_jib_lead: Series[float]
    entry_jib_sheet: Series[float]
    exit_jib_sheet: Series[float]
    entry_jib_sheet_pct: Series[float]
    exit_jib_sheet_pct: Series[float]

    # Cant angles
    entry_cant: Series[float]  # degrees
    exit_cant: Series[float]  # degrees
    cant_drop_target: Series[float]  # degrees
    cant_stow_target: Series[float]  # degrees

    # Performance metrics
    vmg_distance: Series[float]
    distance: Series[float]
    dist_2: Series[float]
    theoretical_vmg: Series[float]
    theoretical_target_vmg: Series[float]
    theoretical_distance: Series[float]
    theoretical_targ_distance: Series[float]
    loss_vs_vmg: Series[float]
    loss_vs_targ_vmg: Series[float]

    # Navigation
    bearing: Series[float]  # degrees
    bearing_2: Series[float]  # degrees
    b_diff: Series[float]
    b_diff_1: Series[float]
    b_diff_2: Series[float]

    # Other
    dashboard: Series[str]
    entry_tack: Series[str]  # port/starboard
    t_invert: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds
    t_to_lock: Series[float] = pa.Field(nullable=True, coerce=True)  # seconds
    drop_offset: Series[float]
    drop_to_wind_axis: Series[float]  # degrees
    htw_bsp: Series[float] = pa.Field(nullable=True, coerce=True)  # km/h
    winward_rh_at_drop: Series[float] = pa.Field(nullable=True, coerce=True)  # mm
