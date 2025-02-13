import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from pandera.typing import DataFrame

from src.models.dataclasses import Boundary, CompoundMark, Mark, Race
from src.models.dataframes import PolarDataSchema


def import_boat_log(file_path: Path | str) -> pd.DataFrame:
    """
    Import a single boat log CSV file and add boat identifier metadata.

    Args:
        file_path: Path to boat log CSV file (e.g., data_GER.csv)

    Returns:
        DataFrame containing boat log with boat identifier
    """
    # Read the CSV file
    if isinstance(file_path, str):
        file_path = Path(file_path)

    df = pd.read_csv(file_path)

    # Extract boat identifier from filename (e.g., "data_GER.csv" -> "GER")
    boat_id = file_path.stem.split("_")[-1]

    # Add metadata columns
    df["boat_id"] = boat_id
    df["source_file"] = file_path.name

    ## Typecasting
    df["TRK_RACE_NUM_unk"] = df["TRK_RACE_NUM_unk"].astype(str)
    df["TRK_LEG_NUM_unk"] = df["TRK_LEG_NUM_unk"].astype(int)
    df["TRK_LEG_NUM_TOT_unk"] = df["TRK_LEG_NUM_TOT_unk"].astype(int)

    # Parse datetime column
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], utc=True)
    df["TIME_LOCAL_unk"] = pd.to_datetime(df["TIME_LOCAL_unk"])

    return df


def import_boat_logs(directory: str = "Data/Boat_Logs") -> pd.DataFrame:
    """
    Import all boat log CSV files from a directory, adding the boat identifier
    from the filename as a column.

    Args:
        directory: Path to directory containing boat log files (e.g., data_GER.csv)

    Returns:
        DataFrame containing all boat logs with boat identifiers

    Example:
        >>> df = import_boat_logs("Data/Boat_Logs")
        >>> df['boat_id'].unique()
        array(['GER', 'FRA', 'GBR', ...])
    """
    # Convert to Path object for better path handling
    data_path = Path(directory)

    # List to store all dataframes
    dfs = []

    # Find all CSV files
    for file_path in data_path.glob("*.csv"):
        df = import_boat_log(file_path)
        dfs.append(df)
        break

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by datetime and boat_id
    combined_df = combined_df.sort_values(["DATETIME", "boat_id"])

    return combined_df


def parse_race_course(xml_path: str) -> Race:
    """
    Parse race XML and return RaceCourse object containing marks and metadata

    Args:
        xml_path: Path to race XML file

    Returns:
        RaceCourse object containing ordered marks and race metadata
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Parse race metadata
    start_time = pd.to_datetime(root.find("RaceStartTime").get("Start"), utc=True)
    race_id = root.find("RaceID").text

    # Parse compound marks
    compound_marks = {}
    for cm in root.find("Course"):
        cm_id = int(cm.get("CompoundMarkID"))
        cm_name = cm.get("Name")
        marks = []

        for mark in cm:
            marks.append(
                Mark(
                    name=mark.get("Name"),
                    lat=float(mark.get("TargetLat")),
                    lon=float(mark.get("TargetLng")),
                    seq_id=int(mark.get("SeqID")),
                )
            )

        compound_marks[cm_id] = CompoundMark(
            id=cm_id,
            name=cm_name,
            marks=marks,
            rounding="",  # Will be filled from sequence
            seq_id=0,  # Will be filled from sequence
            zone_size=50.0,
        )

    # Get mark sequence and rounding directions
    ordered_marks = []
    for corner in root.find("CompoundMarkSequence"):
        cm_id = int(corner.get("CompoundMarkID"))
        cm = compound_marks[cm_id]
        cm.rounding = corner.get("Rounding")
        cm.seq_id = int(corner.get("SeqID"))
        ordered_marks.append(cm)

    # Parse boundaries
    boundaries = []
    for boundary in root.findall(".//CourseLimit"):
        points = []
        for limit in boundary.findall("Limit"):
            lat = float(limit.get("Lat"))
            lon = float(limit.get("Lon"))
            points.append((lat, lon))

        # Only add if there are points
        if points:
            # adding two 00 default as that is what is used in the XML
            color = boundary.get("colour", "000000FF")
            boundaries.append(
                Boundary(
                    points=points,
                    name=boundary.get("name", "Boundary"),
                    color=f"#{color[2:]}",
                    fill=boundary.get("fill", "1") == "1",
                    opacity=0.4 if boundary.get("fill", "1") == "1" else 0.1,
                )
            )
    return Race(
        marks=sorted(ordered_marks, key=lambda x: x.seq_id),
        start_time=start_time,
        race_id=race_id,
        boundaries=boundaries,
    )


def load_polar_data(filepath: str) -> DataFrame[PolarDataSchema]:
    """
    Load polar performance data from CSV file.

    The CSV should have TWA (True Wind Angle) values as rows and TWS (True Wind Speed) values as columns.
    First column contains TWA values, first row contains TWS values.
    Cell values represent boat speed (BSP) in km/h.

    Args:
        filepath: Path to polar data CSV file

    Returns:
        DataFrame with columns TWA, TWS, BSP containing polar performance data
    """
    # Read raw CSV data
    df = pd.read_csv(filepath)

    # Melt the dataframe to get TWA/TWS/BSP format
    df = df.melt(id_vars=["TWA/TWS"], var_name="TWS", value_name="BSP")

    # Rename columns
    df = df.rename(columns={"TWA/TWS": "TWA"})

    # Convert strings to floats
    df["TWA"] = df["TWA"].astype(float)
    df["TWS"] = df["TWS"].astype(float)
    df["BSP"] = df["BSP"].astype(float)

    return df


def import_maneuver_summary(file_path: Path | str) -> pd.DataFrame:
    """
    Import maneuver summary CSV file with proper datetime parsing.

    Args:
        file_path: Path to maneuver summary CSV file

    Returns:
        DataFrame containing maneuver data with properly parsed datetimes
    """
    # Read the CSV file
    if isinstance(file_path, str):
        file_path = Path(file_path)

    df = pd.read_csv(file_path)

    # Parse datetime columns
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="mixed")
    df["TIME_LOCAL_unk"] = pd.to_datetime(df["DATETIME"], format="ISO8601", utc=False)

    # Convert race number to string for consistency
    df["race"] = df["race"].astype(str)

    return df
