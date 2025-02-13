from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Mark:
    """Single mark in the race course"""

    name: str
    lat: float
    lon: float
    seq_id: int
    twd: Optional[float] = None  # True Wind Direction at mark
    tws: Optional[float] = None  # True Wind Speed at mark


@dataclass
class CompoundMark:
    """Group of marks that form a gate or single mark"""

    id: int
    name: str
    marks: List[Mark]
    rounding: str
    seq_id: int
    zone_size: float = 50.0

    @property
    def twd(self) -> Optional[float]:
        """Average wind direction across marks in compound"""
        directions = [m.twd for m in self.marks if m.twd is not None]
        return np.mean(directions) if directions else None

    @property
    def tws(self) -> Optional[float]:
        """Average wind speed across marks in compound"""
        speeds = [m.tws for m in self.marks if m.tws is not None]
        return np.mean(speeds) if speeds else None


@dataclass
class Boundary:
    """Race course boundary points"""

    points: List[Tuple[float, float]]  # List of (lat, lon) points
    name: str
    color: str = "#0000FF"  # Default blue color
    fill: bool = True
    opacity: float = 0.2


@dataclass
class Race:
    """
    Container for race course data and metadata.

    Attributes:
        marks: Ordered list of compound marks defining the course
        start_time: Scheduled race start time
        name: Race name/identifier
        wind_direction: Target wind direction for the course
        wind_speed: Target wind speed for the course
        boundaries: List of course boundaries
    """

    marks: List[CompoundMark]
    start_time: datetime
    race_id: str
    boundaries: List[Boundary] = field(default_factory=list)

    @classmethod
    def from_xml(cls, xml_path: str, wind_data_path: Optional[str] = None) -> "Race":
        """
        Create Race instance from XML and optional wind data

        Args:
            xml_path: Path to race course XML
            wind_data_path: Optional path to wind data CSV

        Returns:
            Race instance with populated mark and wind data
        """
        from src.importer import parse_race_course

        race = parse_race_course(xml_path)

        if wind_data_path and Path(wind_data_path).exists():
            race.attach_wind_data(wind_data_path)

        return race

    def attach_wind_data(self, wind_data_path: str) -> None:
        """
        Attach wind data to individual marks from CSV file

        Args:
            wind_data_path: Path to CSV containing wind readings
        """
        # Load wind data
        wind_df = pd.read_csv(wind_data_path)

        # Ensure required columns exist
        required_cols = ["DATETIME", "MARK", "TWD_deg", "TWS_km_h_1"]
        if not all(col in wind_df.columns for col in required_cols):
            raise ValueError(f"Wind data missing required columns: {required_cols}")

        # Convert datetime if needed
        wind_df["DATETIME"] = pd.to_datetime(wind_df["DATETIME"])

        # Remove data before race start time
        wind_df = wind_df[wind_df["DATETIME"] >= self.start_time]
        # Attach wind data to each mark
        for compound_mark in self.marks:
            for mark in compound_mark.marks:
                # Try to match by source ID first
                mark_wind = wind_df[wind_df["MARK"] == mark.name]
                if not mark_wind.empty:
                    # Calculate mean wind direction and speed
                    mark.twd = mark_wind["TWD_deg"].mean()
                    mark.tws = mark_wind["TWS_km_h_1"].mean()
                else:
                    # Set defaults if no wind data found
                    mark.twd = 0.0
                    mark.tws = 0.0


@dataclass
class PolarPoint:
    """Single point on polar diagram"""

    twa: float  # True wind angle
    tws: float  # True wind speed
    bsp: float  # Boat speed


@dataclass
class RoutePoint:
    """A point along the optimal route"""

    lat: float
    lon: float
    twa: float  # True wind angle
    heading: float  # Boat heading
    speed: float  # Expected boat speed
    distance: float  # Distance from leg start
    maneuver: Optional[str] = None  # 'tack', 'gybe', or None


@dataclass
class LegRoute:
    """Optimal route for a single leg"""

    start_mark: CompoundMark
    end_mark: CompoundMark
    leg_type: str  # 'upwind', 'downwind', 'reaching'
    points: List[RoutePoint]  # Ordered list of route points
    gate_choice: Optional[str] = None  # Which gate mark was chosen
    total_distance: float = 0.0
    maneuver_count: int = 0


@dataclass
class CourseRoute:
    """Complete route around the course"""

    legs: List[LegRoute]
    total_distance: float
    total_maneuvers: int
