from typing import List, Tuple

import branca
import branca.colormap
import folium
import pandas as pd
import pandera as pa
from bokeh.plotting import figure
from pandera.typing import DataFrame

from src.models.dataframes import VMCVisualizationDataSchema

from ..models.dataclasses import Boundary, CompoundMark, Race
from ..utils.gps import filter_gps_jumps


@pa.check_types(lazy=True)
def create_race_map(
    position_data: DataFrame[VMCVisualizationDataSchema] = None, race: Race = None
) -> Tuple[folium.Map, object]:
    """Create an interactive map visualization of the race"""
    course = race.marks
    # Create base map
    center_lat = position_data["LATITUDE_GPS"].mean()
    center_lon = position_data["LONGITUDE_GPS"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, control_scale=True)

    # Add course marks

    # Add boundaries if race object is provided
    if race and race.boundaries:
        _add_boundaries(m, race.boundaries)

    _add_course_marks(m, course)

    # Add boat track
    if position_data is not None:
        _add_boat_track(m, position_data)

    return m


@pa.check_types(lazy=True)
def _add_boat_track(
    m: folium.Map, position_data: DataFrame[VMCVisualizationDataSchema]
):
    """Add boat track to map, colored by VMC values

    Args:
        m: Folium map object to add track to
        position_data: DataFrame with boat position data
    """
    # Filter out any GPS jumps
    # Create color-coded track segments based on VMC
    raw_points = list(
        zip(position_data["LATITUDE_GPS"], position_data["LONGITUDE_GPS"])
    )

    # Filter GPS jumps
    points = raw_points

    # Find valid indices for VMC values after filtering

    # Add start and finish icons using filtered points
    start_point = points[0]
    end_point = points[-1]

    # Add start icon (green flag)
    folium.Marker(
        location=start_point,
        popup="Start",
        icon=folium.Icon(icon="flag", color="green", prefix="fa"),
    ).add_to(m)

    # Add finish icon (checkered flag)
    folium.Marker(
        location=end_point,
        popup="Finish",
        icon=folium.Icon(icon="flag-checkered", color="black", prefix="fa"),
    ).add_to(m)

    # Create colormap for VMC values
    colormap = branca.colormap.LinearColormap(
        colors=["red", "yellow", "green"],
        vmin=0,
        vmax=position_data["VMC_km_h_1"].max(),
        caption="VMC (km/h)",
    )

    # Add track segments using filtered points
    for i in range(len(points) - 1):
        segment = [points[i], points[i + 1]]
        vmc = position_data["VMC_km_h_1"].values[i]

        # Get color from colormap
        color = colormap.rgb_hex_str(vmc)

        folium.PolyLine(
            segment,
            weight=3,
            color=color,
            opacity=0.8,
            tooltip=f'VMC: {vmc:.1f} km/h Boat Speed: {position_data["SPEED"].values[i]} km/h idx: {position_data.index[i]}',
        ).add_to(m)

    # Add colormap to map
    m.add_child(colormap)


def _add_course_marks(m: folium.Map, course: List[CompoundMark]):
    """Add course marks to the map"""
    for cm in course:
        if len(cm.marks) == 2 and cm.name in ["SL1", "FL1"]:
            _add_line_marks(m, cm)
        else:
            _add_single_marks(m, cm)


def _add_line_marks(m: folium.Map, compound_mark: CompoundMark):
    """Add start/finish line marks"""
    folium.PolyLine(
        locations=[
            [compound_mark.marks[0].lat, compound_mark.marks[0].lon],
            [compound_mark.marks[1].lat, compound_mark.marks[1].lon],
        ],
        color="blue",
        weight=2,
        opacity=1.0,
        popup=f"{compound_mark.name} Line",
    ).add_to(m)

    for mark in compound_mark.marks:
        folium.CircleMarker(
            location=[mark.lat, mark.lon],
            radius=5,
            popup=f"{compound_mark.name} - {mark.name}",
            color="blue",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)


def _add_single_marks(m: folium.Map, compound_mark: CompoundMark):
    """Add individual course marks"""
    for mark in compound_mark.marks:
        color = "yellow" if mark.name == "M1" else "red"
        folium.CircleMarker(
            location=[mark.lat, mark.lon],
            radius=5,
            popup=mark.name,
            color=color,
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)


def _add_boundaries(m: folium.Map, boundaries: List[Boundary]):
    """Add course boundaries to the map"""
    # Sort boundaries to put Exclusion Zone first
    # and draw everything on top of it
    boundaries = sorted(
        boundaries, key=lambda x: 0 if x.name == "Exclusion Zone" else 1
    )
    for boundary in boundaries:
        # Create a polygon for the boundary
        folium.Polygon(
            locations=boundary.points,
            color=boundary.color,
            weight=2,
            fill_color=boundary.color,
            fill_opacity=boundary.opacity,
            popup=boundary.name,
        ).add_to(m)


@pa.check_types(lazy=True)
def visualize_vmc(position_data: DataFrame[VMCVisualizationDataSchema], race: Race):
    """
    Create visualizations of the VMC calculations with VMC-colored track
    Returns both a Folium map and matplotlib figure

    Args:
        position_data: DataFrame with boat position and speed data
        race: Race object containing course marks and boundaries

    Returns:
        Tuple[folium.Map, plt.Figure]: Interactive map and time series plot
    """

    # Filter GPS jumps# Filter GPS jumps
    points = list(zip(position_data["LATITUDE_GPS"], position_data["LONGITUDE_GPS"]))
    position_data = position_data[
        pd.Series(points).isin(filter_gps_jumps(points))
    ].copy()

    # Create time series plot with bokeh
    p = figure(width=800, height=300, x_axis_type="datetime", title="Boat Speed vs VMC")

    # Add boat speed line
    p.line(
        position_data.index,
        position_data["SPEED"],
        line_color="blue",
        line_alpha=0.7,
        legend_label="Boat Speed",
    )

    # Add VMC line
    p.line(
        position_data.index,
        position_data["VMC_km_h_1"],
        line_color="red",
        line_alpha=0.7,
        legend_label="VMC",
    )

    # Customize plot
    p.yaxis.axis_label = "Speed (km/h)"
    p.grid.grid_line_alpha = 0.3
    p.legend.location = "top_right"

    # Create map using map_utils
    m = create_race_map(position_data, race)

    # Add start and finish icons
    start_point = (
        position_data["LATITUDE_GPS"].iloc[0],
        position_data["LONGITUDE_GPS"].iloc[0],
    )
    end_point = (
        position_data["LATITUDE_GPS"].iloc[-1],
        position_data["LONGITUDE_GPS"].iloc[-1],
    )

    # Add start icon (green flag)
    folium.Marker(
        location=start_point,
        popup="Start",
        icon=folium.Icon(icon="flag", color="green", prefix="fa"),
    ).add_to(m)

    # Add finish icon (checkered flag)
    folium.Marker(
        location=end_point,
        popup="Finish",
        icon=folium.Icon(icon="flag-checkered", color="black", prefix="fa"),
    ).add_to(m)

    return m, p
