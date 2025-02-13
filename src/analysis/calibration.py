import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime strings from the data files"""
    try:
        # Convert to timezone-naive datetime
        dt = pd.to_datetime(dt_str)
        if dt.tz is not None:
            dt = dt.tz_localize(None)
        return dt
    except:
        return None


class CalibrationAnalyzer:
    def __init__(self):
        self.mark_winds = None
        self.boat_data = None
        self.results = None

        self.column_map = {
            "TWS": ["TWS_SGP_km_h_1", "TWS_km_h_1", "tws", "TWS_kn"],
            "TWD": ["TWD_SGP_deg", "TWD_deg", "avg_TWD"],
            "LAT": ["LATITUDE_GPS_unk", "LATITUDE_deg", "LAT"],
            "LON": ["LONGITUDE_GPS_unk", "LONGITUDE_deg", "LON"],
        }

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns according to standard naming convention"""
        rename_dict = {}
        for std_name, possible_names in self.column_map.items():
            found = False
            for col_name in possible_names:
                if col_name in df.columns:
                    rename_dict[col_name] = std_name
                    found = True
                    break
            if not found:
                print(f"Warning: No matching column found for {std_name}")
                print(f"Available columns: {df.columns.tolist()}")

        if rename_dict:
            df = df.rename(columns=rename_dict)

        # Verify required columns exist after renaming
        required_cols = ["TWS", "TWD"]
        if "MARK" not in df.columns:  # Only check lat/lon for boat data
            required_cols.extend(["LAT", "LON"])

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns after renaming: {missing}")

        return df

    def load_data(self, mark_winds_path: str, boat_logs_path: str):
        """Load mark wind and boat data"""
        # Load mark winds data
        self.mark_winds = pd.read_csv(mark_winds_path)
        self.mark_winds = self._rename_columns(self.mark_winds)
        if "DATETIME" in self.mark_winds.columns:
            self.mark_winds["DATETIME"] = self.mark_winds["DATETIME"].apply(
                parse_datetime
            )
            self.mark_winds.set_index("DATETIME", inplace=True)

        # Load boat data
        self.boat_data = pd.read_csv(boat_logs_path)
        self.boat_data = self._rename_columns(self.boat_data)
        if "DATETIME" in self.boat_data.columns:
            # Ensure timezone-naive datetime
            self.boat_data["DATETIME"] = pd.to_datetime(
                self.boat_data["DATETIME"]
            ).dt.tz_localize(None)

    def _find_nearest_mark_readings(
        self,
        boat_time: datetime,
        boat_lat: float,
        boat_lon: float,
        time_window: timedelta = timedelta(minutes=5),
    ) -> pd.DataFrame:
        """Find mark readings near a boat's position and time"""
        # Ensure boat_time is timezone-naive
        if hasattr(boat_time, "tz") and boat_time.tz is not None:
            boat_time = boat_time.tz_localize(None)

        # Filter mark readings by time window
        time_mask = (self.mark_winds.index >= boat_time - time_window) & (
            self.mark_winds.index <= boat_time + time_window
        )

        # Create a copy to avoid SettingWithCopyWarning
        nearby_readings = self.mark_winds[time_mask].copy()

        if nearby_readings.empty:
            return pd.DataFrame()

        # Calculate distances to marks
        R = 6371  # Earth's radius in km
        lat1, lon1 = np.radians(boat_lat), np.radians(boat_lon)
        lat2, lon2 = np.radians(nearby_readings["LAT"]), np.radians(
            nearby_readings["LON"]
        )

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = R * c

        # Use loc to set values
        nearby_readings.loc[:, "distance"] = distances

        # Return filtered copy
        return nearby_readings[nearby_readings["distance"] < 0.5].copy()  # Within 500m

    def analyze_mark_winds(self) -> pd.DataFrame:
        """Compare boat wind readings with mark wind readings"""
        if self.mark_winds is None or self.boat_data is None:
            raise ValueError("Please load data first using load_data()")

        comparisons = []

        # Sample boat data at regular intervals
        for _, boat_row in self.boat_data.iterrows():
            nearby_marks = self._find_nearest_mark_readings(
                boat_row["DATETIME"], boat_row["LAT"], boat_row["LON"]
            )

            if not nearby_marks.empty:
                for _, mark_row in nearby_marks.iterrows():
                    comparisons.append(
                        {
                            "datetime": boat_row["DATETIME"],
                            "mark": mark_row.get("MARK", "Unknown"),
                            "distance": mark_row["distance"],
                            "boat_tws": boat_row["TWS"],
                            "mark_tws": mark_row["TWS"],
                            "boat_twd": boat_row["TWD"],
                            "mark_twd": mark_row["TWD"],
                            "tws_diff": boat_row["TWS"] - mark_row["TWS"],
                            "twd_diff": np.abs(
                                (boat_row["TWD"] - mark_row["TWD"] + 180) % 360 - 180
                            ),
                        }
                    )

        self.results = pd.DataFrame(comparisons)
        return self.results

    def show_mark_analysis(self):
        """Display mark wind analysis results"""
        if self.results is None:
            raise ValueError("Please run analysis first using analyze_mark_winds()")

        print("\nWind Calibration Analysis:")
        print("--------------------------")

        # Group by mark and calculate statistics
        mark_stats = self.results.groupby("mark").agg(
            {"tws_diff": ["mean", "std", "count"], "twd_diff": ["mean", "std", "count"]}
        )

        print("\nDifferences by Mark (Boat - Mark):")
        print(mark_stats)

        # Overall statistics
        print("\nOverall Statistics:")
        print(
            f"Mean TWS Difference: {self.results['tws_diff'].mean():.2f} ± {self.results['tws_diff'].std():.2f}"
        )
        print(
            f"Mean TWD Difference: {self.results['twd_diff'].mean():.2f} ± {self.results['twd_diff'].std():.2f}"
        )


def validate_vmc_calculations(vmc_data: pd.DataFrame, boat_data: pd.DataFrame) -> None:
    """
    Validate VMC calculations and print statistics
    """
    print("VMC Validation:")
    print("--------------")
    print(f"Total points: {len(vmc_data)}")
    print("VMC Statistics:")
    print(vmc_data["VMC_km_h_1"].describe())

    # Check for impossible VMC values
    impossible_vmc = vmc_data[vmc_data["VMC_km_h_1"] > boat_data["BOAT_SPEED_km_h_1"]]
    if len(impossible_vmc) > 0:
        print(f"\nWARNING: Found {len(impossible_vmc)} points where VMC > boat speed!")

    # Check for mark progression
    mark_changes = vmc_data["next_mark"].ne(vmc_data["next_mark"].shift())
    print(f"\nMark Roundings: {mark_changes.sum()}")

    # Print mark sequence
    print("\nMark Sequence:")
    mark_sequence = vmc_data[mark_changes]["next_mark"].tolist()
    for i, mark in enumerate(mark_sequence, 1):
        print(f"{i}. {mark}")
