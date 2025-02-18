# notebook_utils.py
import os
import numpy as np
import pandas as pd
import glob
from ast import literal_eval
import matplotlib.pyplot as plt
import pickle
import pandapower

# preprocess utils
def load_csv_files(DIRECTORY_PATH):
    """Load all CSV files from the specified directory that match a specific pattern."""
    csv_files = glob.glob(f'{DIRECTORY_PATH}simulations_data_*.csv')
    dfs = []
    for i, file_path in enumerate(csv_files):
        print(f"Loading Simulations Data ID-{i+1} from: {file_path}")
        dfs.append(pd.read_csv(file_path))
    return dfs

def combine_and_save_dataframes(dfs, DIRECTORY_PATH, OUTPUT_CSV_FILE_PATH='combined_solution.csv'):
    combined_df = pd.concat(dfs, ignore_index=True) # Combine DataFrames
    combined_df.to_csv(f'{DIRECTORY_PATH}{OUTPUT_CSV_FILE_PATH}', index=False) # Save to CSV
    print(f'Saved combined DataFrame at: {DIRECTORY_PATH}{OUTPUT_CSV_FILE_PATH}')
    return combined_df

def extract_numeric_data(df, capacities):
    for column in ['transformer_num_violations_dam', 'lines_num_violations_dam_sum']:
        for capacity in capacities:
            # Extract percentage as float, count as int
            df[f'{column}_limit'] = df[capacity][column].str.extract(r"'([\d.]+)%'").astype(float)
            df[capacity][column] = df[capacity][column].str.extract(r'(\d+)').astype(int)
    return df

# Group DataFrame by a column and save each group as a CSV.
def group_and_save(df):
    grouped = df.groupby('MPV_SOLAR_CELL_CAPACITY_WATT')
    reset_group_dfs = {}
    for capacity, group in grouped:
        reset_group = group.reset_index(drop=True)
        reset_group_dfs[capacity] = reset_group
        print(f"Group {capacity}: {len(reset_group)} records.")
        base_path = "output/mcs_results/"
        file_name = f'{capacity}_capacity.csv'
        full_path = base_path + file_name
        group.to_csv(full_path, index=False) # Save group to CSV
        print(f'Saved: {file_name}')
    return reset_group_dfs

def process_capacity_groups(reset_group_dfs, capacities):
    """ Processes groups in 'reset_group_dfs' for each capacity in 'capacities'.
    :param reset_group_dfs: Dictionary of DataFrames, grouped by capacity.
    :param capacities: List of capacities to process.
    :return: The processed DataFrame for the last capacity in the list. """
    for capacity in capacities:
        # Check if the capacity exists in the dictionary
        if capacity in reset_group_dfs:
            # Access the DataFrame for the specific capacity
            group_df = reset_group_dfs[capacity]
            # Convert string representations to actual dictionaries
            group_df['kpi_key'] = group_df['kpi_key'].apply(literal_eval)
            # Extract values from dictionaries and add as new columns
            for key in group_df['kpi_key'].iloc[0].keys():
                reset_group_dfs[capacity][key] = group_df['kpi_key'].apply(lambda x: x.get(key))
            # Output the processed group
            print(f"Group for capacity {capacity}")
            # print(group_df)
        else:
            print(f"Group for capacity {capacity} not found.")
    return reset_group_dfs

def preprocess_data(directory_path, output_csv_file_path):
    """ Load, combine, and preprocess data from CSV files in a specified directory.
    Parameters:
    - directory_path: Path to the directory containing the CSV files.
    - output_csv_file_path: Path to save the combined CSV file.
    Returns:
    - reset_group_dfs: A list of DataFrames, each processed and reset based on unique capacities.
    """
    # Load CSV files
    loaded_dfs = load_csv_files(directory_path)
    
    # Combine and save DataFrames
    combined_df = combine_and_save_dataframes(loaded_dfs, directory_path, output_csv_file_path)
    
    # Determine unique capacities
    capacities = sorted(combined_df['MPV_SOLAR_CELL_CAPACITY_WATT'].unique())
    
    # Sort the combined DataFrame
    sorted_df = combined_df.sort_values(['MPV_CONCENTRATION_RATE_PERCENT', 'MPV_INVERTER_APPARENT_POWER_WATT', 'MPV_SOLAR_CELL_CAPACITY_WATT']).reset_index(drop=True)
    
    # Group by capacity and save each group
    grouped_dfs = group_and_save(sorted_df)
    
    # Process each capacity group
    processed_group_dfs = process_capacity_groups(grouped_dfs, capacities)
    
    # Extract numeric data
    reset_group_dfs = extract_numeric_data(processed_group_dfs, capacities)

    print("Preprocess successful!")
    
    return reset_group_dfs, capacities

# plot utils
def calculate_global_limits(reset_group_dfs, capacities, kpi_param='transformer_loading_avg'):
    """
    Calculates the global minimum and maximum values for both x and y axes across
    multiple dataframes, each representing a different capacity. Adds a 5% margin
    to these limits.
    """
    global_x_min, global_x_max = float('inf'), float('-inf')
    global_y_min, global_y_max = float('inf'), float('-inf')
    for capacity in capacities:
        df = reset_group_dfs[capacity]
        x_min, x_max = df['MPV_CONCENTRATION_RATE_PERCENT'].min(), df['MPV_CONCENTRATION_RATE_PERCENT'].max()
        y_min, y_max = df[kpi_param].min(), df[kpi_param].max()
        global_x_min, global_x_max = min(global_x_min, x_min), max(global_x_max, x_max)
        global_y_min, global_y_max = min(global_y_min, y_min), max(global_y_max, y_max)

    x_range, y_range = global_x_max - global_x_min, global_y_max - global_y_min
    global_x_min -= 0.05 * x_range
    global_x_max += 0.05 * x_range
    global_y_min -= 0.05 * y_range
    global_y_max += 0.05 * y_range
    return global_x_min, global_x_max, global_y_min, global_y_max

def cm_to_inches(cm):
    """ Converts centimeters to inches. """
    return cm / 2.54

def inches_to_cm(inches):
    """ Converts inches to centimeters. Supports both single values and lists/tuples. """
    return [value * 2.54 for value in inches] if isinstance(inches, (list, tuple)) else inches * 2.54

# Function to create a row of plots for a specific KPI parameter
def create_combined_plot(reset_group_dfs, CAPACITIES, axes, 
                         row_index, kpi_param, y_label, GLOBAL_X_MIN, GLOBAL_X_MAX, GLOBAL_Y_MIN, GLOBAL_Y_MAX, marker_styles):
    for i, capacity in enumerate(CAPACITIES):
        df = reset_group_dfs[capacity]
        # Iterate over each power value and plot
        for power, style in marker_styles.items():
            marker_style = style[0]
            marker_size = style[1]
            df_filtered = df[df['MPV_INVERTER_APPARENT_POWER_WATT'] == power]
            label = f'$\\gamma_{{1}}$ = {power} VA'
            # Plotting the filtered data
            axes[row_index, i].plot(df_filtered['MPV_CONCENTRATION_RATE_PERCENT'], df_filtered[kpi_param], marker_style, markersize=marker_size, label=label)
        # Setting X and Y limits for the plot
        axes[row_index, i].set_xlim(GLOBAL_X_MIN, GLOBAL_X_MAX)
        axes[row_index, i].set_ylim(GLOBAL_Y_MIN, GLOBAL_Y_MAX)
        # Setting Y-axis label for the first subplot in each row
        if i == 0:
            axes[row_index, i].set_ylabel(y_label)
        else:
            axes[row_index, i].tick_params(axis='y', which='both', labelleft=False)

def calc_and_round_setpoints(data_list, num_setpoints=5, decimal_places=3, 
                             small_step_indices=None, middle_step_indices=None, 
                             small_middle_indices=None, small_step_sizes=[0.01, 1, 2, 5], 
                             middle_step_sizes=[5, 10, 15, 20], 
                             small_middle_sizes=[3, 6, 9, 12], 
                             large_step_sizes=[25, 50, 75], repetitions=4):
    # Default settings
    if small_step_indices is None:
        small_step_indices = []
    if middle_step_indices is None:
        middle_step_indices = []
    if small_middle_indices is None:
        small_middle_indices = []

    result_list = []
    for index, item in enumerate(data_list):
        start = item[2]
        end = item[3]
        # Repetitions for each set of setpoints
        for _ in range(repetitions):
            rounded_setpoints = []
            # Calculate setpoints
            for i in range(num_setpoints):
                value = start + i * (end - start) / (num_setpoints - 1)
                # Adjust first and last values to avoid edge clipping
                if i == 0:
                    value -= abs(value * 0.00001)
                elif i == num_setpoints - 1:
                    value += abs(value * 0.00001)

                # Choose appropriate step size based on index
                if index in small_step_indices:
                    step_sizes = small_step_sizes
                elif index in middle_step_indices:
                    step_sizes = middle_step_sizes
                elif index in small_middle_indices:
                    step_sizes = small_middle_sizes
                else:
                    step_sizes = large_step_sizes

                # Round to the nearest step size
                rounded_value = round(value / step_sizes[0]) * step_sizes[0]

                # Make sure the value is even after the decimal point
                fraction_part = rounded_value - int(rounded_value)
                if (fraction_part * (10 ** decimal_places)) % 2 != 0:
                    rounded_value += step_sizes[0] / (10 ** decimal_places)

                # Avoid repeat values
                if rounded_setpoints and rounded_value <= rounded_setpoints[-1]:
                    rounded_value = rounded_setpoints[-1] + step_sizes[0]

                rounded_setpoints.append(rounded_value)

            # Append the set of rounded setpoints
            result_list.append(rounded_setpoints)

    return result_list

# Basic configurations for half DIN A4 page
marker_size = 3
plot_configurations = [
    {
        "page": "Configuration for half-page plot",
        "dimensions_cm": (21, 14.85 - 3), # Width, Height (caption space already subtracted)
        "spacing": (0.8, 0.15), # hspace, wspace
        "font_size": 9,
        "y_labels": ['VM (p.u.)', 'GL (MW)', 'TL (%)', 'LL (%)'],
        "x_label": r'$\beta \ (\%)$',
        "X_TICKS": [0, 25, 50, 75, 100],
        "marker_styles": {600: ('^:', marker_size), 800: ('s--', marker_size), 1000: ('D-.', marker_size)},
        "kpi_params": ['vm_lv_mean_max', 'grid_loss_p_mw', 'transformer_loading_avg', 'lines_loading_mean_max']
    }
]
# Steps and sizes for axes
step_indices_sizes = {
    "small_step": ([0], [0.0002, 0.0005, 0.004, 0.002, 0.02, 0.2, 0.5]),
    "small_middle": ([1], [0.5, 1, 2]),
    "middle_step": ([2], [1, 2, 5]),
    "large_step": ([], [1, 2, 5]) # Example for empty indices, if needed
}

def plot_analysis(reset_group_dfs, CAPACITIES, file_name, base_path = "output/notebook_cs_1/"):
    # Plot creation
    for config in plot_configurations:
        fig_width, fig_height = cm_to_inches(config["dimensions_cm"][0]), cm_to_inches(config["dimensions_cm"][1])
        hspace, wspace = config["spacing"]
        print(f"Output: {config['page']}\nCreating plot with width: {fig_width} inches, height: {fig_height} inches")
    
        # Corrected potential issue: Ensure that the subplot dimensions are integers
        fig, axes = plt.subplots(4, 4, figsize=(fig_width, fig_height))
        globals_xy = [calculate_global_limits(reset_group_dfs, CAPACITIES, kpi_param=kpi) for kpi in config['kpi_params']]
        # Correctly unpack the dictionary into named arguments
        y_ticks_list = calc_and_round_setpoints(
            data_list=globals_xy,
            small_step_indices=step_indices_sizes["small_step"][0],
            small_middle_indices=step_indices_sizes["small_middle"][0],
            middle_step_indices=step_indices_sizes["middle_step"][0],
            small_step_sizes=step_indices_sizes["small_step"][1],
            small_middle_sizes=step_indices_sizes["small_middle"][1],
            middle_step_sizes=step_indices_sizes["middle_step"][1],
            large_step_sizes=step_indices_sizes["large_step"][1])
        for row_index, kpi_param in enumerate(config['kpi_params']):
            create_combined_plot(reset_group_dfs, CAPACITIES, axes, row_index, kpi_param, config['y_labels'][row_index],
                                 *globals_xy[row_index], config['marker_styles'])
    
        for i, ax in enumerate(axes.flatten()):
            ax.set_xlabel(config['x_label'])
            ax.grid(True)
            ax.set_xticks(config['X_TICKS'])
            ax.set_yticks(y_ticks_list[i])
    
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        # Create legend and adjust marker styles in the legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.50, 0.95), ncol=len(labels), frameon=False)

        # Adjustments before saving and displaying the plot
        full_path_pdf = base_path + file_name + '.pdf'
        full_path_pgf = base_path + file_name + '.pgf'

        # Save the plot before calling plt.show()
        plt.savefig(full_path_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(full_path_pgf, format='pgf', bbox_inches='tight')
        print(f"The plot has been saved as {full_path_pdf}")

        # Now display the plot
        plt.show()
    return fig, axes

def utils_scan_folders(path, start_name_criteria):
    """ Scans a directory for subdirectories that start with specific criteria.
    Args:
        path (str): The path to the directory to scan.
        start_name_criteria (list of str): Criteria that the directory names should start with.
    Returns:
        list of str: A list of directory names that match the criteria.
    """
    scanned_folders = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and any(name.startswith(start_name) for start_name in start_name_criteria):
            scanned_folders.append(name)
    return scanned_folders

def utils_sort_folders_by_desired_order(desired_order, scanned_folders):
    """ Sorts folders based on a desired order, ensuring unique entries.
    Args:
        desired_order (list of str): The desired order for the folder names.
        scanned_folders (list of str): The folders that were scanned.
    Returns:
        list of str: The folders sorted based on the desired order.
    """
    sorted_folders = []
    seen = set()
    for criteria in desired_order:
        for folder in scanned_folders:
            if criteria in folder and folder not in seen:
                sorted_folders.append(folder)
                seen.add(folder)
    return sorted_folders

def utils_load_data_from_folders(path, sorted_folders):
    """ Loads pickle files from a list of folders.
    Args:
        path (str): The base path where the folders are located.
        sorted_folders (list of str): The folders from which to load the data.
    Returns:
        dict: A dictionary with the folder names as keys and the loaded data as values.
    """
    raw_data = {}
    for folder in sorted_folders:
        pickle_file_path = os.path.join(path, folder, 'log_variables.pickle')
        try:
            with open(pickle_file_path, 'rb') as file:
                raw_data[folder] = pickle.load(file)
                print(f"Data loaded from: {pickle_file_path}")
        except FileNotFoundError:
            print(f"File not found: {pickle_file_path}")
        except Exception as e:
            print(f"Error loading file {pickle_file_path}: {e}")
    return raw_data

