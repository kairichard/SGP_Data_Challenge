from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

import numpy as np
import pandas as pd
from ipywidgets import interact, FloatSlider, fixed
import pandera as pa
from pandera.typing import DataFrame
from src.models.dataframes import CompassDataSchema
from src.navigation.compass import calculate_compass_average

@pa.check_types
def plot_averaged_twd(target_rate: float, compass_data: DataFrame[CompassDataSchema]) -> None:
    """
    Plot original vs averaged TWD data, breaking the line at large time gaps.
    """
    data = pd.DataFrame({
        'timestamp': pd.to_datetime(compass_data['DATETIME']),
        'direction': compass_data['DIRECTION']
    })

    # Calculate window size in seconds
    window_size = int(1/target_rate)
    
    # Calculate time differences
    time_diffs = data['timestamp'].diff().dt.total_seconds()
    
    # Find indices where there are large gaps (e.g., > 2 seconds)
    gap_indices = np.where(time_diffs > 2)[0]
    
    # Create the plot
    p = figure(width=800, height=400, x_axis_type="datetime",
              title=f'Direction: Original vs {1/target_rate:.1f}s Average')
    
    # Plot original data
    source = ColumnDataSource(data)
    p.scatter('timestamp', 'direction', size=2, color='blue', alpha=0.5, 
            legend_label='Original TWD', source=source)
    
    # Plot averaged data in segments
    start_idx = 0
    for end_idx in gap_indices:
        # Get segment of data
        segment = data.iloc[start_idx:end_idx]
        
        if len(segment) > window_size:  # Only process if segment is large enough
            # Calculate averaged TWD for this segment
            averaged_twd = calculate_compass_average(
                directions=segment['direction'].values,
                sample_rate=1,
                target_rate=target_rate
            )
            
            # Get timestamps for averaged data
            averaged_timestamps = segment['timestamp'].iloc[::window_size].values[:len(averaged_twd)]
            
            # Plot this segment as a separate line to avoid connecting across gaps
            source = ColumnDataSource({
                'timestamp': averaged_timestamps,
                'direction': averaged_twd
            })
            p.line('timestamp', 'direction', line_color='red', line_width=2,
                  source=source, line_join='bevel')
            
            start_idx = end_idx+1
    
    # Process the last segment
    last_segment = data.iloc[start_idx:]
    if len(last_segment) > window_size:
        averaged_twd = calculate_compass_average(
            directions=last_segment['direction'].values,
            sample_rate=1,
            target_rate=target_rate
        )
        averaged_timestamps = last_segment['timestamp'].iloc[::window_size].values[:len(averaged_twd)]
        source = ColumnDataSource({
            'timestamp': averaged_timestamps,
            'direction': averaged_twd
        })
        p.line('timestamp', 'direction', line_color='red', line_width=2,
              legend_label=f'{1/target_rate:.1f}s Average Direction', source=source)
    
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'True Wind Direction (degreessss)'
    p.legend.location = "top_right"
    p.grid.grid_line_alpha = 0.3
    
    # Print statistics
    stats = f"""
    Averaging window: {1/target_rate:.1f} seconds
    Time range: {data['timestamp'].min()} to {data['timestamp'].max()}
    """
    print(stats)
    
    show(p)

# Create interactive widget
def plot_averaged_twd_widget(compass_data: DataFrame[CompassDataSchema]):
    CompassDataSchema.validate(compass_data)
    return interact(
        plot_averaged_twd,
        target_rate=FloatSlider(
            min=0.01,
            max=.20,
            step=0.01,
            value=0.1,
            description='Rate (Hz):',
            continuous_update=False
        ),
        compass_data=fixed(compass_data)
)