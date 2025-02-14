from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pandera as pa
from pandera.typing import DataFrame
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.models.dataframes import ManeuverDataSchema
from src.transformations.maneuver_processor import ManeuverProcessor


class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple methods"""

    def __init__(self, processor: ManeuverProcessor):
        self.processor = processor
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.feature_importance_results = {}

    @pa.check_types
    def fit(self, df: DataFrame[ManeuverDataSchema]) -> None:
        """
        Fit models and calculate feature importance

        Args:
            df: Raw maneuver data
        """
        # Process data
        X, y = self.processor.fit_transform(df)

        # Fit models
        self.rf_model.fit(X, y)
        self.gb_model.fit(X, y)

        # Calculate feature importances
        self._calculate_random_forest_importance(X)
        self._calculate_gradient_boosting_importance(X)
        self._calculate_shap_importance(X)

    def _calculate_random_forest_importance(self, X: pd.DataFrame) -> None:
        """Calculate Random Forest feature importance"""
        importance = pd.DataFrame(
            {"feature": X.columns, "importance": self.rf_model.feature_importances_}
        )
        self.feature_importance_results["random_forest"] = importance.sort_values(
            "importance", ascending=False
        )

    def _calculate_gradient_boosting_importance(self, X: pd.DataFrame) -> None:
        """Calculate Gradient Boosting feature importance"""
        importance = pd.DataFrame(
            {"feature": X.columns, "importance": self.gb_model.feature_importances_}
        )
        self.feature_importance_results["gradient_boosting"] = importance.sort_values(
            "importance", ascending=False
        )

    def _calculate_shap_importance(self, X: pd.DataFrame) -> None:
        """Calculate SHAP feature importance"""
        explainer = shap.TreeExplainer(self.rf_model)
        shap_values = explainer.shap_values(X)

        importance = pd.DataFrame(
            {"feature": X.columns, "importance": np.abs(shap_values).mean(axis=0)}
        )
        self.feature_importance_results["shap"] = importance.sort_values(
            "importance", ascending=False
        )

    def plot_feature_importance(self, top_n: int = 10) -> None:
        """
        Plot feature importance from Random Forest method

        Args:
            top_n: Number of top features to display
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Feature Importance Analysis", fontsize=16)

        # Plot Random Forest importance
        self._plot_importance(
            self.feature_importance_results["random_forest"],
            "Random Forest Feature Importance",
            ax,
            top_n,
        )

        return fig

    def _plot_importance(
        self, importance_df: pd.DataFrame, title: str, ax: plt.Axes, top_n: int
    ) -> None:
        """Helper function to plot feature importance"""
        top_features = importance_df.head(top_n)

        sns.barplot(data=top_features, x="importance", y="feature", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")

    def get_top_features(self, top_n: int = 10) -> Dict[str, List[str]]:
        """
        Get top features from each method

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary with top features for each method
        """
        return {
            method: list(importance_df.head(top_n)["feature"])
            for method, importance_df in self.feature_importance_results.items()
        }

    def print_top_features(self, top_n: int = 10) -> None:
        """Print top features from each method"""
        top_features = self.get_top_features(top_n)

        print("Top Features by Method:")
        print("-" * 40)

        for method, features in top_features.items():
            print(f"\n{method.replace('_', ' ').title()}:")
            for i, feature in enumerate(features, 1):
                print(f"{i}. {feature}")
