# MultiDataPlotting

`MultiDataPlotting` is a Python package designed for easy and efficient plotting of multiple data sets, providing tools for generating bar plots, scatter plots, and other types of visual data representations. It leverages popular libraries such as Matplotlib, Seaborn, and Pandas to offer a versatile set of plotting functionalities.

## Features

- Plot multiple data sets in a single figure.
- Support for various types of plots: bar plots, scatter plots, line plots, rose maps.
- Customizable plot settings: colors, labels, legends, and more.
- Integration with Pandas for direct data frame plotting.

## Installation

You can install `MultiDataPlotting` using pip:

```bash
pip install MultiDataPlotting
```

## Usage

To use the package, you can import it into your Python scripts as follows:

```python
import multidataprocessing as mdp
```

### Plotting Time Histograms

The `plot_time_histogram` function creates histograms based on time data, ideal for visualizing event frequencies over specific intervals. This method helps in identifying trends and patterns by illustrating how event occurrences are distributed across different time points.

_Functionality_

This function is particularly useful in scenarios such as analyzing user activity logs, sales data over time, or any periodic data that is timestamped. It helps in understanding the distribution and concentration of events, aiding in decision-making processes like resource allocation during peak times

_Parameters_

- **histogram** (dict): Dictionary with datetime keys (either strings or objects) and frequency counts as values.
- **color** (str, optional): Color of the histogram bars. Default is 'blue'.
- **edgecolor** (str, optional): Color of the bar edges. Default is 'black'.
- **fig_size** (tuple, optional): Size of the figure in inches. Default is (10, 6).
- **tick_fontname** (str, optional): Font name for the tick labels. Default is 'Arial'.
- **tick_fontsize** (int, optional): Font size for the tick labels. Default is 12.
- **title_fontsize** (int, optional): Font size for the title of the plot. Default is 14.
- **label_fontsize** (int, optional): Font size for the axis labels. Default is 14.
- **y_range** (list, optional): List specifying the y-axis limits as [min, max]. If not set, the axis scales automatically.
- **x_tick_interval_minutes**: (int,optional): the interval of the x ticks in minutes
- **x_ticklabel_format** (str, optional): Format of the x-tick labels, can be 'HH' for hours and minutes, or 'YYYY-MM-DD' for dates. Default is 'HH'.
- **is_legend** (bool, optional): Indicates whether to display a legend. Default is False.
- **save_path** (str, optional): File path where the plot will be saved, if desired. If not provided, the plot is not saved.
- **is_show** (bool, optional): Whether to display the plot on the screen. Default is False.
- **is_save** (bool, optional): Whether to save the plot to a file. Default is True.
- **transparent_bg** (bool, optional): Whether the background of the saved figure should be transparent. Default is True.

_Code Example_

```python
import multidataplotting as mdp

# Example histogram data
# Example usage with datetime conversion
login_data = {
    '00:00:00 - 00:03:00': 100,
    '00:03:00 - 00:06:00': 150,
    '00:07:00 - 00:12:00': 125,
    '00:19:00 - 00:23:00': 175,
    '00:29:00 - 00:33:00': 175,
}

mdp.plot_time_histogram(login_data, color='blue', edgecolor='black', is_show=True)
```
![alt text](https://github.com/wwang487/MultiDataPlotting/blob/main/picture/timehist.png?raw=true)

### Plotting Polylines

The `plot_polylines` function in the multidataprocessing package is designed to plot multiple lines from a DataFrame using specified columns for the x and y axes. This functionality is essential for visualizing trends and relationships in data over time or across categories.

_Functionality_

This function is invaluable for comparing trends across multiple datasets, enabling clear insights into how different variables evolve concurrently. It is particularly useful in financial analysis, environmental monitoring, or any scientific data visualization involving multiple variables.

_Parameters_

- **df** (pandas.DataFrame): The DataFrame containing the data to plot.
- **x** (int): Index of the column in the DataFrame to use as the x-axis.
- **ys** (list of int): Indices of columns to plot on the y-axis.
- **line_styles** (dict, optional): Dictionary specifying line styles for each y-column plotted, with keys as column indices and values as style strings. Default is solid lines for all.
- **line_widths** (dict, optional): Dictionary specifying line widths for each plotted y-column, with keys as column indices and values as numeric widths. Default is 2 for all lines.
- **line_colors** (dict, optional): Dictionary specifying colors for each plotted y-column, with keys as column indices and values as color names or hex codes. Default is 'blue' for all lines.
- **legends** (list, optional): List of legend labels corresponding to each y-column. If not provided, column names are used as labels.
- **marker_colors** (dict, optional): Dictionary specifying marker colors for each y-column, enhancing distinctiveness of each line.
- **figsize** (tuple, optional): Size of the figure in inches. Default is (10, 6).
- **x_tick_interval** (int, optional): Interval at which x-axis ticks should be placed. Default is 1.
- **markers** (dict, optional): Dictionary specifying marker styles for each y-column plotted, adding visual distinction to the lines.
- **y_label** (str, optional): Label for the y-axis. If not provided, defaults to a generic label.
- **show_grid** (bool, optional): Whether to display a grid in the background of the plot. Default is True.
- **font_name** (str, optional): Font family for all text elements in the plot. Default is 'Arial'.
- **font_size** (int, optional): Font size for all text elements in the plot. Default is 12.
- **save_path** (str, optional): Path to save the figure to a file. If not specified, the figure is not saved.
- **dpi** (int, optional): Resolution of the saved figure in dots per inch. Default is 600.
- **y_range** (list, optional): Specific limits for the y-axis as [min, max].
- **is_show** (bool, optional): Whether to display the plot on the screen. Default is True.

_Code Example_

```python
import pandas as pd
import numpy as np
import multidataplotting as mdp

data = {
    'X': list(range(12)),
    'Product1': np.random.randint(1000, 5000, size=12),
    'Product2': np.random.randint(1000, 5000, size=12)
}

data = pd.DataFrame(data)
mdp.plot_polylines(data, x=0, ys=[1, 2], line_colors={1: 'red', 2: 'green'}, x_tick_interval = 2, markers={1: 'o', 2: 'x'})
```
![alt text](https://github.com/wwang487/MultiDataPlotting/blob/main/picture/polyline.png?raw=true)

### Plotting Histograms

The `plot_histograms` function from the multidataprocessing package allows users to plot multiple histograms in a single figure. This feature is particularly useful for comparing the distributions of several datasets simultaneously, providing insights into variability, skewness, or similarity among the datasets.

_Functionality_

`plot_histograms` is ideal for statistical analysis and data exploration, helping to visualize and compare the frequency distributions of multiple groups or variables within the same graphical context. It's especially useful in fields like biostatistics, market research, and any area where distribution comparison is crucial.

_Parameters_

- **list_of_lists** (list of lists): A list where each sublist contains numeric data for one histogram.
- **titles** (list of str): Titles for each histogram. Must match the number of sublists in list_of_lists.
- **xlabels** (str or list of str, optional): X-axis labels for each histogram. Can be a single string or a list of strings matching the number of histograms.
- **ylabels** (str or list of str, optional): Y-axis labels for each histogram. Can be a single string or a list of strings matching the number of histograms.
- **bins** (int, optional): Number of bins for the histograms. Default is 10.
- **color** (str, optional): Color of the histogram bars. Default is 'blue'.
- **edgecolor** (str, optional): Color of the bar edges. Default is 'black'.
- **fig_size** (tuple, optional): Size of the figure, in inches. Default is (10, 6).
- **tick_fontname** (str, optional): Font name for the tick labels. Default is 'Arial'.
- **tick_fontsize** (int, optional): Font size for the tick labels. Default is 12.
- **title_fontsize** (int, optional): Font size for the titles of each plot. Default is 14.
- **label_fontsize** (int, optional): Font size for the labels. Default is 14.
- **value_range** (list, optional): Specific range of values to include in each histogram as [min, max].
- **line_color** (str, optional): Color of the mean line within each histogram. Default is 'red'.
- **show_all_xticklabels** (bool, optional): Whether to show all x-axis tick labels. Default is True.
- **line_style** (str, optional): Style of the mean line, such as '--' for dashed lines. Default is '--'.
- **line_width** (float, optional): Width of the mean line. Default is 2.
- **is_legend** (bool, optional): Whether to include a legend in the plots. Default is False.
- **unit** (str, optional): Unit of measurement for the data, useful for annotations. Default is 'm'.
- **is_log** (bool, optional): Whether to use a logarithmic scale for the y-axis. Default is False.
- **is_log_x** (bool, optional): Whether to use a logarithmic scale for the x-axis. Default is False.
- **is_fixed_y_range** (bool, optional): Whether to fix the y-axis range across all plots. Default is False.
- **y_range** (list, optional): Specific y-axis limits as [min, max] if is_fixed_y_range is True.
- **is_mean_value** (bool, optional): Whether to display the mean value on the histograms. Default is True.
- **is_scaled** (bool, optional): Whether to scale the data by a factor. Default is False.
- **scale_factor** (float, optional): Factor by which to scale the data if is_scaled is True. Default is 10.
- **save_path** (str, optional): File path where the plot will be saved, if desired.
- **is_show** (bool, optional): Whether to display the plot on the screen. Default is True.
- **is_save** (bool, optional): Whether to save the plot to a file. Default is True.
- **transparent_bg** (bool, optional): Whether the background of the saved figure should be transparent. Default is True.
- **hspace** (float, optional): Space between subplots, in normalized subplot coordinates. Default is 0.1.
- **label_sep** (str, optional): Separator for splitting the labels. Default is '_'.

_Code Example_

```python
import numpy as np
import multidataplotting as mdp

# Generate random age data for four groups
group1 = np.random.normal(loc=30, scale=5, size=100)  # Mean age 30, SD 5
group2 = np.random.normal(loc=40, scale=5, size=100)  # Mean age 40, SD 5
group3 = np.random.normal(loc=20, scale=5, size=100)  # Mean age 20, SD 5
group4 = np.random.normal(loc=50, scale=5, size=100)  # Mean age 50, SD 5
list_of_lists = [group1, group2, group3, group4]

# Titles for each histogram
titles = ["Group 1: 20s", "Group 2: 30s", "Group 3: 40s", "Group 4: 50s"]
xlabels = ['A', 'B', 'C', 'D']
ylabels = 'Frequency (%)'
# Plotting histograms
mdp.plot_histograms(list_of_lists, titles=titles, xlabels = None, ylabels = ylabels, is_show=True, hspace = 0.2, color = '#edf8b1')
```
![alt text](https://github.com/wwang487/MultiDataPlotting/blob/main/picture/hists.png?raw=true)

### Drawing Categorical Bar and Curve Plots

The `draw_cat_bar_curveplots` function from the multidataprocessing package combines categorical bar plots with overlaid curve plots. This feature allows for a detailed comparative analysis of primary data alongside secondary trends within the same visual context, making it ideal for simultaneous examination of discrete and continuous data aspects.

_Functionality_

This function is perfect for applications where understanding the relationship between categorical and continuous data is essential. It can be used in market research for product comparison across different time points, financial analysis showing categorical expenditure against revenue trends, or any scenario where integrated visualization provides deeper insights.

_Parameters_

- **main_result** (dict): Main categorical data to be plotted as bars. Keys should be categories and values the measurements.
- **other_data_list** (list of dicts): List of dictionaries for the curve plots, where each dictionary represents a dataset to be plotted as a curve over the bar plots.
- **bar_colors** (list, optional): List of colors for the bars in the bar plot. If not provided, a default set of colors is used.
- **bar_thickness** (float, optional): Thickness of the bars. Default is 0.8.
- **bar_edge_color** (str, optional): Color of the bar edges. Default is 'black'.
- **line_color** (str, optional): Color for all line plots. Default is 'black'.
- **cat_labels** (list, optional): the labels for each category
- **xlabels** (list, optional): the labels for the x-axis
- **ylabels** (list, optional): the labels for the y-axis
- **y_range** (list, optional): The range for the y-axis as [min, max]. If not set, the axis scales automatically.
- **figsize** (tuple, optional): Size of the figure, in inches. Default is (10, 6).
- **line_thickness** (float or list, optional): Thickness of the lines for curve plots. Can be a single value or a list of values corresponding to each dataset.
- **tick_fontsize** (int, optional): Font size for the tick labels. Default is 10.
- **tick_fontname** (str, optional): Font name for the tick labels. Default is 'sans-serif'.
- **x_tick_interval** (int, optional): Interval between x-ticks, useful for densely packed x-axes. Default is 1.
- **is_show** (bool, optional): Whether to display the plot on the screen. Default is False.
- **is_save** (bool, optional): Whether to save the plot to a file. Default is True.
- **save_path** (str, optional): File path where the plot will be saved, if desired. If not specified, the plot is not saved.

_Code Example_

```python
import multidataplotting as mdp

# Main results for the bar plot
main_result = {'1993-01-01': [200, 10], '1993-01-03': [10, 240], '1993-01-05': [300, 100], '1993-01-06': [100, 120]}

# Additional data for curve plots
other_data_list = [
    {'1993-01-01': 150, '1993-01-02': 180, '1993-01-04': 210, '1993-01-07': 160},  # Product A
    {'1993-01-01': 100, '1993-01-03': 200, '1993-01-05': 230}   # Product B
]

# Plotting the combined bar and curve plots
mdp.draw_cat_bar_curveplots(
    main_result,
    other_data_list,
    bar_colors=['blue', 'green', 'red'],
    line_color='black',
    figsize=(12, 8),
    tick_fontsize=12,
    is_show=True,
    cat_labels=['A', 'A+'],
    xlabels = ['', '', 'Date'],
    ylabels = ['Data 1', 'Data 2', 'Data 3']
)
```
![alt text](https://github.com/wwang487/MultiDataPlotting/blob/main/picture/cat_curve.png?raw=true)

### Plotting Bar Plots

The `plot_bar_plots` function from the multidataprocessing package is designed to plot multiple bar plots within a single figure, allowing for comparative analysis across different data sets or categories. This function is particularly useful for visualizing and comparing discrete data points across multiple groups or time intervals.

_Functionality_

This function is ideal for business reporting, scientific data comparison, or any scenario where multiple datasets need to be compared side-by-side. It enhances data presentation by providing clear, distinct visualizations that facilitate easy comparison of categorical data.

_Parameters_

- **list_of_lists** (list of lists): A list where each sublist contains data for one bar plot.
- **tuple_range_list** (list of tuples): A list of tuples specifying the range for each bar in the plots.
- **titles** (str or list of str, optional): Titles for each subplot. Can be a single string or a list of strings if there are multiple plots.
- **ylabels** (str or list of str, optional): Y-axis labels for each subplot. Can be a single string or a list of strings if there are multiple plots.
- **bar_color** (str, optional): Color of the bars in the plots. Default is 'blue'.
- **bar_edgecolor** (str, optional): Color of the bar edges. Default is 'black'.
- **fig_size** (tuple, optional): Dimensions of the entire figure, in inches. Default is (10, 6).
- **tick_fontname** (str, optional): Font name for the tick labels. Default is 'Arial'.
- **tick_fontsize** (int, optional): Font size for the tick labels. Default is 12.
- **title_fontsize** (int, optional): Font size for the titles of each subplot. Default is 14.
- **label_fontsize** (int, optional): Font size for the labels. Default is 14.
- **line_color** (str, optional): Color of the line indicating mean or median, if applicable. Default is 'red'.
- **show_all_xticklabels** (bool, optional): Whether to show all x-tick labels. Default is True.
- **bar_width** (float, optional): Width of the bars in the plots. Default is 1.
- **line_style** (str, optional): Style of the mean or median line, e.g., '--' for dashed. Default is '--'.
- **line_width** (float, optional): Width of the mean or median line. Default is 2.
- **is_legend** (bool, optional): Whether to show a legend in the plots. Default is False.
- **unit** (str, optional): Unit of measurement for the data, useful for annotations. Default is 'm'.
- **is_fixed_y_range** (bool, optional): Whether to fix the y-axis range across all plots. Default is True.
- **y_range** (list, optional): Specific y-axis limits as [min, max].
- **is_mean_value** (bool, optional): Whether to display the mean value on the histograms. Default is False.
- **is_scaled** (bool, optional): Whether to scale the data by a factor. Default is False.
- **scale_factor** (float, optional): Factor by which to scale the data, if is_scaled is True.
- **save_path** (str, optional): File path where the plot will be saved, if desired.
- **is_show** (bool, optional): Whether to display the plot on the screen. Default is False.
- **is_save** (bool, optional): Whether to save the plot to a file. Default is True.
- **transparent_bg** (bool, optional): Whether the background of the saved figure should be transparent. Default is True.
- **horizontal** (bool, optional): Whether the bars should be plotted horizontally. Default is False.
- **convert_minute** (bool, optional): Whether to convert the x-tick labels to minutes. Default is True.
- **hspace** (float, optional): The amount of height reserved for white space between subplots, expressed as a fraction of the average axis height. Default is 0.05.

_Code Example_

```python
import multidataplotting as mdp
import numpy as np

# Sample data
team_performance = [np.random.randint(50, 100, size=4) for _ in range(3)]
range_tuples = [(1, 3), (4, 6), (7, 9), (10, 12)]  # Quarterly ranges

# Plotting the bar plots
mdp.plot_bar_plots(
    list_of_lists=team_performance,
    tuple_range_list=range_tuples,
    titles=["Team A", "Team B", "Team C"],
    ylabels="Score",
    bar_color='green',
    fig_size=(12, 8),
    is_show=True
)
```

### Plotting 2D Heatmaps

The `plot_2D_heatmap` function in the multidataprocessing package generates a two-dimensional heatmap from a dictionary of tuple pairs representing coordinate points and their respective frequency or intensity values. This visualization tool is especially useful for representing the density or intensity distribution across a two-dimensional space, making it applicable in areas like geographical data analysis, heat distribution, market density studies, and more.

_Functionality_

This function transforms raw data into a visually intuitive heatmap, allowing for the easy identification of patterns or concentrations within the data. It is invaluable for any analysis where understanding the spatial distribution of a variable is crucial.

_Parameters_

- **pair_freq** (dict): Dictionary with keys as tuple pairs of x and y coordinates (e.g., (x, y)) and values as their frequencies or intensities. Or, a dataframe with two columns.
- **x_bin_ticks**(list, optional): a list of numbers, must locate within data range, if input is a dictionary, this is required, optional only if your input is a dataframe
- **y_bin_ticks**(list, optional): a list of numbers, must locate within data range, if input is a dictionary, this is required, optional only if your input is a dataframe
- **fig_size** (tuple, optional): The dimensions of the figure, in inches. Default is (10, 8).
- **title** (str, optional): The title of the heatmap. Default is 'Cross Relationship Heatmap'.
- **title_fontsize** (int, optional): Font size for the title. Default is 16.
- **xlabel** (str, optional): Label for the x-axis. Default is 'Variable 1'.
- **ylabel** (str, optional): Label for the y-axis. Default is 'Variable 2'.
- **label_fontsize** (int, optional): Font size for the axis labels. Default is 14.
- **tick_fontsize** (int, optional): Font size for the tick labels. Default is 12.
- **vmin** (float, optional): Minimum data value that corresponds to the colormap’s lower limit. If None, defaults to the data’s min value.
- **vmax** (float, optional): Maximum data value that corresponds to the colormap’s upper limit. If None, defaults to the data’s max value.
- **cmap** (str, optional): Colormap used for the heatmap. Default is 'viridis'.
- **cbar_label** (str, optional): Label for the colorbar. Default is 'Frequency'.
- **save_path** (str, optional): Path to save the figure. If provided, the plot is saved to this location.
- **is_show** (bool, optional): Whether to display the plot. Default is True.
- **xtick_rotation** (int, optional): Rotation angle for the x-axis tick labels. Default is 0.
- **ytick_rotation** (int, optional): Rotation angle for the y-axis tick labels. Default is 0.
- **xticklabels** (list, optional): Custom labels for the x-axis ticks. If None, default labels based on the data are used.
- **yticklabels** (list, optional): Custom labels for the y-axis ticks. If None, default labels based on the data are used.

_Code Example_

```python
import multidataplotting as mdp

# Example data: coordinates with frequencies
data = {
    (0, 0): 100, (0, 2): 70,
    (0, 1): 150, (1, 2): 205,
    (1, 0): 200, (2, 1): 160,
    (1, 1): 250, (2, 0): 95,
}
x_bin_ticks = [0, 1, 2]
y_bin_ticks = [0, 1, 2]
# Plotting the heatmap
plot_2D_heatmap(pair_freq=data, x_bin_ticks=x_bin_ticks, y_bin_ticks=y_bin_ticks,title="Event Frequency Distribution", 
                xlabel="X", ylabel="Y", is_show=True, is_annotate=True)
```

![alt text](https://github.com/wwang487/MultiDataPlotting/blob/main/picture/heatmap.png?raw=true)

### Plotting Contour Maps

The `plot_contour_map` function in the multidataprocessing package generates contour maps based on spatial data, suitable for visualizing gradients, potential fields, or any continuous variable across a two-dimensional area. This function is extremely useful in geophysical studies, meteorology, and landscape ecology to understand topographical, temperature, or pressure variations in a given region.

_Functionality_

Contour maps created by this function help in interpreting complex spatial data by showing lines of equal value, making it easier to see hills, valleys, and plains in geographical data or gradients in any scientific measurements across a surface.

_Parameters_

- **data** (dict or pandas.DataFrame): Data to plot, which should be a dictionary with tuple pairs (x, y) as keys and values representing the measurement at those coordinates, or a DataFrame with columns for x, y, and value.
- **fig_size** (tuple, optional): Dimensions of the figure, typically in inches. Default is (10, 8).
- **title** (str, optional): Title of the contour map. Default is 'Contour Map'.
- **title_fontsize** (int, optional): Font size for the title. Default is 16.
- **xlabel** (str, optional): Label for the x-axis. Default is 'X Coordinate'.
- **ylabel** (str, optional): Label for the y-axis. Default is 'Y Coordinate'.
- **colorbar_label** (str, optional): Label for the colorbar that describes the variable. Default is 'Frequency (%)'.
- **label_fontsize** (int, optional): Font size for the axis labels. Default is 14.
- **tick_fontsize** (int, optional): Font size for the tick labels. Default is 12.
- **contour_levels** (int, optional): Number of contour levels to draw. Default is 20.
- **cmap** (str, optional): Colormap used for the contour plot. Default is 'viridis'.
- **is_contour_line** (bool, optional): Whether to draw contour lines. Default is True.
- **contour_line_colors** (str, optional): Color of the contour lines. Default is 'white'.
- **contour_line_width** (float, optional): Width of the contour lines. Default is 1.0.
- **is_contour_label** (bool, optional): Whether to label the contour lines. Default is True.
- **contour_label_color** (str, optional): Color of the contour labels. Default is 'white'.
- **contour_label_size** (int, optional): Font size of the contour labels. Default is 10.
- **label_interval** (int, optional): Interval at which labels appear on contour lines. Default is 2.
- **save_path** (str, optional): Path to save the figure. If provided, the plot is saved to this location.
- **is_show** (bool, optional): Whether to display the plot. Default is True.
- **xticks** (list, optional): Custom tick intervals for the x-axis.
- **yticks** (list, optional): Custom tick intervals for the y-axis.
- **xtick_labels** (list, optional): Custom labels for the x-axis ticks.
- **ytick_labels** (list, optional): Custom labels for the y-axis ticks.

_Code Example_

```python
import multidataplotting as mdp
import pandas as pd

# Example Data
data = {
    (0, 0): 100, (0, 2): 70,
    (0, 1): 150, (1, 2): 205,
    (1, 0): 200, (2, 1): 160,
    (1, 1): 250, (2, 0): 95,
}

# if using pandas
# data = pd.DataFrame({'Points': [(0,0),(0,1), (0,2),(1,0),(1,1),(1,2),(2,1),(2,0)], 'Elevation': [100, 150, 70, 200, 250, 205, 160, 95]})
# Plotting the contour map
mdp.plot_contour_map(data, title="Elevation Contour Map", xlabel="Longitude", ylabel="Latitude", colorbar_label="Elevation (m)", is_show=True)
```
![alt text](https://github.com/wwang487/MultiDataPlotting/blob/main/picture/contour.png?raw=true)

### Plotting 3-D Bar Plots

The `plot_3d_stacked_bar` function in the multidataprocessing package creates a three-dimensional stacked bar plot, which is particularly effective for visualizing the distribution of multiple variables across two dimensions with an additional dimension represented by the height of the bars. This type of visualization is useful in finance, sales data analysis, resource allocation, and more, where comparisons across two categorical dimensions are necessary.

_Functionality_

This function enhances data presentation by allowing for the visualization of complex interrelationships among data in a three-dimensional space, making it easier to discern patterns and trends that may not be as obvious in two-dimensional plots.

_Parameters_

- **pair_freq** (dict or pandas.DataFrame): Data to plot, either as a dictionary with tuple pairs (x, y) as keys and values as their frequency, or a DataFrame with suitable columns.
- **x_bin_ticks** (list): Bins or categories for the x-axis.
- **y_bin_ticks** (list): Bins or categories for the y-axis.
- **fig_size** (tuple, optional): Dimensions of the figure, typically in inches. Default is (12, 10).
- **title** (str, optional): Title of the 3D plot. Default is '3D Stacked Bar Plot'.
- **title_fontsize** (int, optional): Font size for the title. Default is 16.
- **xlabel** (str, optional): Label for the x-axis. Default is 'Variable 1'.
- **ylabel** (str, optional): Label for the y-axis. Default is 'Variable 2'.
- **zlabel** (str, optional): Label for the z-axis, representing the frequency or another measured variable. Default is 'Frequency'.
- **is_percent** (bool, optional): Whether to display the z-axis values as percentages. Default is False.
- **label_fontsize** (int, optional): Font size for the axis labels. Default is 10.
- **tick_fontsize** (int, optional): Font size for the tick labels. Default is 10.
- **color_map** (matplotlib.colors.Colormap, optional): Color map for the bars, affecting their color based on value. Default uses Matplotlib's viridis.
- **save_path** (str, optional): Path to save the figure. If provided, the plot is saved to this location.
- **is_show** (bool, optional): Whether to display the plot. Default is True.
- **x_ticklabel_left_close** (bool, optional): Whether to close the left side of x-tick labels. Default is False.
- **x_ticklabel_right_close** (bool, optional): Whether to close the right side of x-tick labels. Default is False.
- **y_ticklabel_top_close** (bool, optional): Whether to close the top side of y-tick labels. Default is False.
- **y_ticklabel_bottom_close** (bool, optional): Whether to close the bottom side of y-tick labels. Default is False.
- **elevation** (int, optional): Elevation angle of the 3D view. Default is 30.
- **azimuth** (int, optional): Azimuth angle of the 3D view. Default is 30.
- **background_color** (str, optional): Background color of the plot. Default is 'white'.

_Code Example_

```python
import multidataplotting as mdp

# Example data: Sales volume for different regions and product categories
data = {
    (0, 0): 100, (0, 2): 70,
    (0, 1): 150, (1, 2): 205,
    (1, 0): 200, (2, 1): 160,
    (1, 1): 250, (2, 0): 95,
}

# Defining the categories for the axes
x_categories = ["Reg 1", "Reg 2"]
y_categories = ["Prod A", "Prod B"]

# Plotting the 3D stacked bar chart
mdp.plot_3d_stacked_bar(data, x_bin_ticks=x_categories, y_bin_ticks=y_categories, \
    elevation = 65, azimuth = 120, ytick_rotation=60,
        title="Annual Sales Volume", xlabel="Region", ylabel="Product Category", zlabel="Sales Volume", is_show=True)
```
![alt text](https://github.com/wwang487/MultiDataPlotting/blob/main/picture/3DStackbar.png?raw=true)

### Plotting Rose Map

The `plot_rose_map` function in the multidataprocessing package creates a rose diagram (or wind rose), which is commonly used in meteorology to display the frequency of occurrence of events relative to multiple directions. This visualization is also highly effective in geography, environmental science, and any field requiring the analysis of directional data.

_Functionality_

This function facilitates the analysis of how often events occur from different directions and can include an additional dimension representing magnitude (such as wind speed) or frequency. It's perfect for visualizing how wind direction and speed vary at a particular location or how any directional data is distributed.

_Parameters_

- **input_data** (pd.DataFrame): The DataFrame containing the data. It should include columns for direction (degrees or category) and the value or frequency of occurrences.
- **key_1** (str): Column name in input_data for directional data.
- **key_2** (str): Column name in input_data for the values associated with each direction.
- **interval** (int, optional): The bin size for directional data, in degrees. Default is 10.
- **value_interval** (int, optional): The bin size for value data, providing granularity in how values are segmented within the plot. Default is None, which means no binning.
- **title** (str, optional): Title of the rose map. Default is "Rose Map".
- **label_size** (int, optional): Font size for the labels on the plot. Default is 12.
- **label_interval** (int, optional): Interval for displaying labels around the rose, which can help in reducing label clutter. Default is 1.
- **color_ramp** (str, optional): Color map used for the plot. Default is "viridis".
- **tick_label_size** (int, optional): Font size for the tick labels. Default is 12.
- **tick_label_color** (str, optional): Color for the tick labels. Default is 'black'.
- **tick_font_name** (str, optional): Font name for the tick labels. Default is 'Arial'.
- **figsize** (tuple, optional): Dimensions of the plot, in inches. Default is (10, 8).
- **colorbar_label** (str, optional): Label for the color bar. Default is "Intensity".
- **colorbar_label_size** (int, optional): Font size for the color bar label. Default is 12.
- **max_radius** (float, optional): Maximum radius for the plot, controlling how far out the data extends. Default is None, which auto-scales.
- **save_path** (str, optional): File path where the plot will be saved, if desired.
- **is_show** (bool, optional): Whether to display the plot. Default is True.

_Code Example_

```python
import pandas as pd
import multidataplotting as mdp

# Example wind data
data = {
    'Direction': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    'Speed': [5, 10, 15, 20, 15, 10, 5, 10, 15, 20, 25, 30]
}
df = pd.DataFrame(data)

# Plotting the wind rose
mdp.plot_rose_map(df, key_1='Direction', key_2='Speed', interval=30, title="Wind Rose Diagram", color_ramp='plasma', is_show=True)
```
![alt text](https://github.com/wwang487/MultiDataPlotting/blob/main/picture/rosemap.png?raw=true)

### Plotting Rose Contour Map for Directional Data

The `plot_rose_contour_map` function creates a rose contour map to visualize directional data using contours with optional threshold-based boundary lines. This is particularly useful in meteorology, oceanography, and environmental sciences where wind, waves, or any directional data need to be visualized for analysis and decision-making.

**Functionality**

This function plots a rose (or wind) contour map with options for customization such as color ramps, boundary lines at specified density thresholds, and percentage labels. It supports various contour levels, providing flexibility for detailed and high-level data representation.

**Parameters**

- **input_data** (pd.DataFrame): The DataFrame containing the directional data.
- **key_1** (str): The column name for directional data.
- **key_2** (str): The column name for the value data.
- **title** (str): Title of the plot.
- **label_size** (int): Font size for labels.
- **color_ramp** (str): Color ramp for the plot.
- **figsize** (tuple): Size of the figure.
- **num_levels** (int): Number of contour levels.
- **max_radius** (float): Maximum radius of the plot.
- **density_threshold** (float): Density threshold for boundary line.
- **z_label** (str): Label for the colorbar.
- **boundary_line_color** (str): Color for the boundary line.
- **boundary_line_thickness** (float): Thickness of the boundary line.
- **is_percent** (bool): Whether to show the colorbar as a percentage.
- **tick_spacing** (int): Spacing between ticks on the colorbar.
- **save_path** (str): Path to save the plot.
- **is_show** (bool): Whether to display the plot.

**Example Usage**

Suppose you are analyzing wind data and need to visualize the distribution and intensity of wind directions:

```python
import pandas as pd
import multidataplotting as mdp
data = pd.DataFrame({
    'direction': np.random.randint(0, 360, 1000),
    'speed': np.random.rand(1000) * 100
})
mdp.plot_rose_contour_map(data, 'direction', 'speed', color_ramp='plasma', num_levels=10, density_threshold=0.001,
                      boundary_line_color='red', boundary_line_thickness=3, is_percent = True)
```
![alt text](https://github.com/wwang487/MultiDataPlotting/blob/main/picture/rosecontourmap.png?raw=true)
