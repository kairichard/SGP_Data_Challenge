"""
Distance to Leader Visualization Module.

This module provides visualization tools for DTL analysis, extending the existing
map visualization functionality with DTL-specific features.

Key Functions:
- visualize_dtl(): Create combined DTL visualizations
- plot_dtl_timeseries(): Plot DTL over time
- create_dtl_map(): Extend race map with DTL information

Visualizations include:
1. Color-coded boat tracks showing DTL
2. Time series plots of DTL for all boats
3. Leader position and transitions
4. Mark rounding validations
5. Interactive elements for analysis

Builds on:
- Existing map_utils functionality
- Matplotlib/Folium integration
- GPS track visualization
"""

from typing import Tuple

import matplotlib.pyplot as plt
import pandera as pa
from bokeh.layouts import column
from bokeh.plotting import figure
from pandera.typing import DataFrame

from src.models.dataframes import DTLVisualizationDataSchema


@pa.check_types(lazy=True)
def visualize_dtl(
    position_data: DataFrame[DTLVisualizationDataSchema],
) -> Tuple[plt.Figure]:
    """
    Create DTL visualizations including map and time series plots.

    Args:
        position_data: Dictionary mapping boat IDs to their position/data DataFrames
        course: List of course marks defining the race course

    Returns:
        Tuple containing:
        - folium.Map: Interactive map with DTL visualization
        - plt.Figure: Time series plot of DTL metrics
    """

    # Define a color palette for boats
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    boat_colors = {
        boat_id: colors[i % len(colors)]
        for i, boat_id in enumerate(position_data.keys())
    }

    # Create time series plots with bokeh
    p1 = figure(
        width=1200,
        height=400,
        x_axis_type="datetime",
        title="Course Distance to Leader",
    )
    p2 = figure(
        width=1200, height=400, x_axis_type="datetime", title="Race Rankings Over Time"
    )

    # Plot DTL course distance
    for boat_id, dtl_df in position_data.items():
        p1.line(
            dtl_df["DATETIME"],
            dtl_df["DTL_DIRECT"],
            legend_label=boat_id,
            alpha=0.7,
            line_color=boat_colors[boat_id],
        )

    p1.yaxis.axis_label = "Distance to Leader (meters)"
    p1.grid.grid_line_alpha = 0.3
    p1.legend.location = "top_right"

    # Calculate rankings at each timestamp
    timestamps = sorted(
        list(set([ts for df in position_data.values() for ts in df["DATETIME"]]))
    )
    rankings = {}

    for ts in timestamps:
        # Get all boats' positions at this timestamp
        positions = []
        for boat_id, df in position_data.items():
            if any(df["DATETIME"] == ts):
                row = df[df["DATETIME"] == ts].iloc[0]
                positions.append((boat_id, row["DTL_DIRECT"], row["LEG_BEHIND"]))

        # Sort by leg first, then by DTL
        positions.sort(key=lambda x: (x[2], x[1]))
        # Assign rankings
        for rank, (boat_id, _, _) in enumerate(positions, 1):
            if boat_id not in rankings:
                rankings[boat_id] = []
            rankings[boat_id].append((ts, rank))

    # Plot rankings
    for boat_id, rank_data in rankings.items():
        times, ranks = zip(*rank_data)
        p2.line(
            times,
            ranks,
            legend_label=boat_id,
            alpha=0.7,
            line_width=2,
            line_color=boat_colors[boat_id],
        )

    p2.yaxis.axis_label = "Race Position"
    p2.y_range.flipped = True  # Invert y-axis
    p2.grid.grid_line_alpha = 0.3
    p2.legend.location = "top_right"

    # Combine plots into layout
    plot = column(p1, p2)

    return plot
