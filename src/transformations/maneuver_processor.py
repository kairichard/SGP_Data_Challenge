from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.dataframes import ManeuverDataSchema


class ManeuverProcessor:
    """Process maneuver data according to analysis plan"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.column_transformer = None

    def _get_feature_groups(self) -> dict:
        """Define feature groups for analysis"""
        return {
            "entry_conditions": [
                "entry_bsp",
                "entry_twa",
                "entry_rh",
                "entry_heel",
                "entry_pitch",
                "entry_cant",
                "entry_jib_lead",
                "entry_jib_sheet",
                "entry_jib_sheet_pct",
            ],
            "exit_conditions": [
                "exit_bsp",
                "exit_twa",
                "exit_rh",
                "exit_heel",
                "exit_pitch",
                "exit_cant",
                "exit_jib_lead",
                "exit_jib_sheet",
                "exit_jib_sheet_pct",
            ],
            "environmental": ["tws", "avg_TWD"],
            "maneuver_execution": [
                "turning_time",
                "max_yaw_rate",
                "db_down",
                "t_swap",
                "max_lat_gforce",
                "max_fwd_gforce",
                "max_rudder_angle",
            ],
            "board_control": [
                "db_ud_ret_press_s_avg",
                "db_ud_ret_press_s_max",
                "db_ud_ret_press_p_avg",
                "db_ud_ret_press_p_max",
                "db_ud_ext_press_s_avg",
                "db_ud_ext_press_s_max",
                "db_ud_ext_press_p_avg",
                "db_ud_ext_press_p_max",
                "db_cant_ret_press_p_avg",
                "db_cant_ret_press_p_max",
                "db_cant_ret_press_s_avg",
                "db_cant_ret_press_s_max",
                "db_cant_ext_press_p_avg",
                "db_cant_ext_press_p_max",
                "db_cant_ext_press_s_avg",
                "db_cant_ext_press_s_max",
            ],
            "flight_control": ["turn_min_rh", "winward_rh_at_drop"],
        }

    def _get_columns_to_drop(self) -> list:
        """Define columns to remove from analysis"""
        return [
            # Identifiers
            "BOAT",
            "HULL",
            # Timestamps
            "DATETIME",
            "TIME_LOCAL_unk",
            # Post-maneuver calculations
            "theoretical_vmg",
            "theoretical_target_vmg",
            "theoretical_distance",
            "theoretical_targ_distance",
            "loss_vs_vmg",
            "loss_vs_targ_vmg",
            # Config settings
            "dashboard",
            "WING_CONFIG_unk",
            "MD4_SEL_DB_unk",
            "MD4_SEL_RUD_unk",
        ]

    def _get_categorical_columns(self) -> list:
        """Define categorical columns for encoding"""
        return ["type", "entry_tack", "flying"]

    def _get_numeric_columns(self) -> list:
        """Get all numeric columns for scaling"""
        feature_groups = self._get_feature_groups()
        numeric_cols = []
        for group in feature_groups.values():
            numeric_cols.extend(group)
        return numeric_cols

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process maneuver data for analysis

        Args:
            df: Raw maneuver data following ManeuverDataSchema

        Returns:
            X: Processed feature matrix
            y: Target variable (vmg_distance)
        """
        # Validate input
        ManeuverDataSchema.validate(df)

        # Split features and target
        y = df["vmg_distance"]
        X = df.drop(columns=["vmg_distance"])

        # Remove irrelevant columns
        X = X.drop(columns=self._get_columns_to_drop())

        # Handle missing values separately for numeric and categorical columns
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = X.select_dtypes(
            include=["object", "string", "category"]
        ).columns

        # Fill numeric missing values with mean
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        # Fill categorical missing values with mode
        X[categorical_cols] = X[categorical_cols].fillna(
            X[categorical_cols].mode().iloc[0]
        )

        # Setup column transformer
        numeric_features = self._get_numeric_columns()
        categorical_features = self._get_categorical_columns()

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        # Fit and transform
        X_transformed = self.column_transformer.fit_transform(X)

        # Get feature names after transformation
        numeric_cols = numeric_features
        categorical_cols = []
        for i, feature in enumerate(categorical_features):
            cats = self.column_transformer.named_transformers_["cat"].categories_[i]
            categorical_cols.extend([f"{feature}_{cat}" for cat in cats])

        # Convert to dataframe with proper column names
        X_transformed = pd.DataFrame(
            X_transformed, columns=numeric_cols + categorical_cols, index=X.index
        )

        return X_transformed, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Transform new data using fitted preprocessor"""
        if self.column_transformer is None:
            raise ValueError("Processor not fitted. Call fit_transform first.")

        # Validate input
        ManeuverDataSchema.validate(df)

        # Split features and target
        y = df["loss_vs_targ_vmg"]
        X = df.drop(columns=["loss_vs_targ_vmg"])

        # Remove irrelevant columns
        X = X.drop(columns=self._get_columns_to_drop())

        # Handle missing values separately for numeric and categorical columns
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = X.select_dtypes(
            include=["object", "string", "category"]
        ).columns

        # Fill numeric missing values with mean
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        # Fill categorical missing values with mode
        X[categorical_cols] = X[categorical_cols].fillna(
            X[categorical_cols].mode().iloc[0]
        )

        # Transform
        X_transformed = self.column_transformer.transform(X)

        # Get feature names
        numeric_features = self._get_numeric_columns()
        categorical_features = self._get_categorical_columns()

        numeric_cols = numeric_features
        categorical_cols = []
        for i, feature in enumerate(categorical_features):
            cats = self.column_transformer.named_transformers_["cat"].categories_[i]
            categorical_cols.extend([f"{feature}_{cat}" for cat in cats])

        # Convert to dataframe
        X_transformed = pd.DataFrame(
            X_transformed, columns=numeric_cols + categorical_cols, index=X.index
        )

        return X_transformed, y
