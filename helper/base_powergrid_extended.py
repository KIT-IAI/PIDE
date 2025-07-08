"""
base_powergrid_extended Module
------------------------------
This module extends the functionality of grid storage control in Pandapower. It includes classes
for implementing various control strategies for grid storage elements and analyzing time series
simulations. The module is built upon the `BasePowerGridCore` class and provides additional
capabilities for storage control and evaluation.
"""

# Importing necessary libraries
import os
import json
import pickle
import numpy as np
import pandas as pd

# Importing parent module
from .base_powergrid_core import BasePowerGridCore


class JSONEncoder(json.JSONEncoder):
    """A custom JSON encoder that supports converting objects with a 'to_json'
    method to JSON format."""

    def default(self, o):
        if hasattr(o, "to_json"):
            return o.to_json(orient="records")
        return json.JSONEncoder.default(self, o)


class BasePowerGridExtended(BasePowerGridCore):
    """
    A class (BasePowerGridExtended) representing a power grid reinforcement
    learning strategy. This class inherits from the BasePowerGridCore and adds
    attributes and methods specific to reinforcement learning strategies.
    The reinforcement learning algorithm is expected to interact with this
    class through its public methods.
    Attributes:
    (1): Joint Parameters
        args (dict): The input arguments for the simulation.
        main_powergrid (pandapowerNet): A copy of the main power grid.
        resolution (int): The time resolution for the simulation.
        episode_start_hour (int): The start hour of the simulation episode.
        episode_start_day (int): The start day of the simulation episode.
        episode_start_min_interval (int): The start interval of the simulation episode.
        episode_limit (int): The limit for the simulation episode.
        net (pandapowerNet): The power grid for the simulation.
        profiles (dict): Data profiles for power grid elements/units.
        log_variables (dict): The log variables for the simulation.
        column_sheet_mapping (list): The column sheet mapping for the simulation.
        output (dict): The results of the simulation in data frame format.
        file_name (str): The name of the file.
        file_path (str): The path of the file.
        base_sim_steps (list): The base time series timestamp data from the originals.
        sim_steps (list): The selected time series range normalized starting with zero up to
            the maximum duration of the simulation.
        max_steps (int): The maximum number of time steps.
        lv_bus_mask (pandas.Series): A mask to get the indices of all the low-voltage (LV) buses
            in the grid.
        lv_buses_index (pandas.Index): The indexing for the low-voltage power grid.
        simulation_mode (str): The simulation mode.
    (2)-Rule-Based-Control (RBC) PandaPower Time Series Simulation-
        output_writer (OutputWriter): The instance of the output writer for the simulation.
    (3)-Rule-Based-Control (RBC) Manual Time Series Simulation-
        storage (dict): The energy storage for the simulation.
        renderer (object): An instance of the renderer class used by PyQT5 to display
            the simulation results.
        rendering_flag (bool): The flag used to determine if rendering is enabled or not.
        bus_line_conditions (tuple): The overloaded line and violated bus indices with empty list.
    (4)-Reinforcement-Learning-Control (RLC) Time Series Simulation-
        steps (int): The number of simulation steps.
        sum_rewards (float): The sum of rewards.
        history (int): The observation history.
        obs_history (dict): The observation history for the simulation.
        pv_histories (list): The PV histories for the simulation.
        active_demand_histories (list): The active demand histories for the simulation.
        reactive_demand_histories (list): The reactive demand histories for the simulation.
        active_load_histories (list): The active load histories for the simulation.
        reactive_load_histories (list): The reactive load histories for the simulation.
        voltage_barrier_type (str): The voltage barrier type.
        voltage_weight (float): The weight for the voltage.
        q_weight (float): The weight for the Q-value.
        line_weight (float): The weight for the line.
        dv_dq_weight (float): The weight for the dV/dQ.
        action_space (gym.spaces): The action space for the simulation.
        state_space (gym.spaces): The state space for the simulation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (1): Joint Parameters
        # Reset power grid and initialize variables
        self.net = self.base_powergrid
        self.args = self.args
        self.profiles = {}
        self.log_variables = {}
        self.file_name_log_variables = "log_variables"
        self.column_sheet_mapping = []
        self.output = {}
        self.file_name, self.file_path = "", ""
        self.resolution = self.resolution
        self.base_sim_steps, self.sim_steps, self.max_steps = None, None, None
        # Identify low-voltage (LV) buses in the grid
        self.lv_bus_mask = self.net.bus.vn_kv < 1.0
        self.lv_buses_index = self.net.bus.loc[self.lv_bus_mask].index
        self.simulation_mode = None
        # self.mpv_range_index = self.mpv_range_index
        # Episode start parameters
        self.episode_start_hour = self.episode_start_hour
        self.episode_start_day = self.episode_start_day
        self.episode_start_min_interval = self.episode_start_min_interval
        self.episode_limit = self.episode_limit

        self.first_date_simulation = None
        self.last_date_simulation = None
        # (2)-Rule-Based-Control (RBC) PandaPower Time Series Simulation-
        self.output_writer = None  # Save OutputWriter instance
        self.variables_to_log = None
        self.variables_to_store = None

        # (3)-Rule-Based-Control (RBC) Manual Time Series Simulation-
        # Initialize variables
        self.storage = {}
        self.renderer = None
        self.rendering_flag = False
        self.bus_line_conditions = []

        # (4)-Reinforcement-Learning-Control (RLC) Time Series Simulation-
        # Initialize variables
        self.steps, self.sum_rewards = 1, 0
        self.history = getattr(self.args["rlc"], "history", 1)
        self.obs_history = {}

        self.pv_histories = None
        self.active_demand_histories = None
        self.reactive_demand_histories = None
        self.active_load_histories = None
        self.reactive_load_histories = None

        self.voltage_barrier_type = None
        self.voltage_weight = None
        self.q_weight = None
        self.line_weight = None
        self.dv_dq_weight = None

        self.action_space, self.state_space = None, None

    def store_logs(self):
        """
        Store simulation results into output data frames.
        Store the logged variables in a dictionary to be written to file.
        This function stores simulation results into output data frames,
        following a pre-defined mapping of log variables to column sheet
        names and column names.
            The variables are stored in pandas dataframes with specific column
            names, which are mapped to the specific variables.
            The dataframes are then stored in a larger dictionary with specific
            keys that correspond to the specific variables.
        This larger dictionary is used to write the dataframes to file using the
        write_output() function.
        """
        # Definition of the default log variables
        self.column_sheet_mapping = [
            (
                "load_active [MW] (res_load.p_mw)",
                "res_load.p_mw",
                self.net.load.index.to_list(),
            ),
            (
                "load_reactive [MVar] (res_load.q_mvar)",
                "res_load.q_mvar",
                self.net.load.index.to_list(),
            ),
            ("pv_active [MW] (sgen.p_mw)", "sgen.p_mw", self.net.sgen.index.to_list()),
            (
                "pv_reactive [MVar] (sgen.q_mvar)",
                "sgen.q_mvar",
                self.net.sgen.index.to_list(),
            ),
            (
                "bus_active_demand [MW] (res_bus.p_mw)",
                "res_bus.p_mw",
                self.net.bus.index.to_list(),
            ),
            (
                "bus_reactive_demand [Mvar] (res_bus.q_mvar)",
                "res_bus.q_mvar",
                self.net.bus.index.to_list(),
            ),
            (
                "bus_voltage [p.u] (res_bus.vm_pu)",
                "res_bus.vm_pu",
                self.net.bus.index.to_list(),
            ),
            (
                "bus_voltage_angle [degree] (res_bus.va_degree)",
                "res_bus.va_degree",
                self.net.bus.index.to_list(),
            ),
            (
                "ext_grid_active_supply [MW] (res_ext_grid.p_mw)",
                "res_ext_grid.p_mw",
                self.net.ext_grid.index.to_list(),
            ),
            (
                "ext_grid_reactive_supply [MVar] (res_ext_grid.q_mvar)",
                "res_ext_grid.q_mvar",
                self.net.ext_grid.index.to_list(),
            ),
            (
                "line_loss [MW] (res_line.pl_mw)",
                "res_line.pl_mw",
                self.net.line.index.to_list(),
            ),
            (
                "line_loss [MVar] (res_line.ql_mvar)",
                "res_line.ql_mvar",
                self.net.line.index.to_list(),
            ),
            (
                "line_loading [%] (res_line.loading_percent)",
                "res_line.loading_percent",
                self.net.line.index.to_list(),
            ),
            (
                "trafo loading (res_trafo.loading_percent)",
                "res_trafo.loading_percent",
                self.net.trafo.index.to_list(),
            ),
            (
                "trafo_active [MW] (res_trafo.p_hv_mw)",
                "res_trafo.p_hv_mw",
                self.net.trafo.index.to_list(),
            ),
            (
                "trafo_reactive [Mvar] (res_trafo.q_hv_mvar)",
                "res_trafo.q_hv_mvar",
                self.net.trafo.index.to_list(),
            ),
            (
                "trafo_active power losses [MW] (res_trafo.pl_mw)",
                "res_trafo.pl_mw",
                self.net.trafo.index.to_list(),
            ),
            (
                "trafo_reactive power losses [MVar] (res_trafo.ql_mvar)",
                "res_trafo.ql_mvar",
                self.net.trafo.index.to_list(),
            ),
        ]
        # Add the log variables for the storage case
        if self.is_storage_scenario:
            optional_storage_column_sheet_mapping = [
                (
                    "storage_active (storage.p_mw)",
                    "storage.p_mw",
                    self.net.storage.index.to_list(),
                ),
                (
                    "storage_reactive (storage.q_mvar)",
                    "storage.q_mvar",
                    self.net.storage.index.to_list(),
                ),
                (
                    "storage_soc_percent (storage.soc_percent)",
                    "storage.soc_percent",
                    self.net.storage.index.to_list(),
                ),
                (
                    "storage_discharge_power_loss (storage.discharge_power_loss)",
                    "storage.discharge_power_loss",
                    self.net.storage.index.to_list(),
                ),
            ]
            self.column_sheet_mapping.extend(optional_storage_column_sheet_mapping)
        # Iterate over the mapping list and create the output
        for log_description, log_name, element_ids in self.column_sheet_mapping:
            if self.simulation_mode == "pandapower":
                data_frame = (
                    self.log_variables[log_name]
                    if isinstance(self.log_variables[log_name], pd.DataFrame)
                    else pd.DataFrame(self.log_variables[log_name])
                )
            else:
                data_frame = pd.DataFrame(self.log_variables[log_name])
                data_frame.columns = element_ids
            self.output[log_name] = data_frame

    def write_to_disk(self):
        """Writes the output stored in `self.output` to disk.
        The function saves the output to disk in multiple file formats, including
        JSON, Excel, and pickle.
        The file format is determined by the `output_file_type` attribute of the
        `Output` object.
        """
        # Define log variable filename
        self.file_name = os.path.join(
            self.output_data_path, self.file_name_log_variables
        )  # absolute path
        if self.output_data_path is not None:
            try:
                if self.output_file_type in [
                    ".csv",
                    ".xls",
                    ".xlsx",
                    ".json",
                    ".p",
                    ".pickle",
                ]:
                    self.file_path = os.path.join(
                        self.file_name + self.output_file_type
                    )
                else:
                    raise UserWarning(
                        "\n----------Write to Disk (1)----------\n"
                        " Specify output file with "
                        " .csv, .xls, .xlsx, .p or .json ending"
                    )
            except ValueError as error:
                print(
                    "\n----------Write to Disk (1)----------\n"
                    f"Error occurred: {error}"
                )
        # Save in loop for all output file types
        output_file_types = [".pickle", ".xlsx", ".json"]
        print("\n----------Write to Disk (2)----------\n")
        print("log_variables output_file_path:        \n")
        for output_file_type in output_file_types:
            # self.output_file_type = output_file_type
            self.file_path = os.path.join(
                self.output_data_path, self.file_name_log_variables + output_file_type
            )
            print(f"'{self.file_path}'")
            # Saves all parameters as object attributes to store in JSON
            if output_file_type == ".json":
                with open(self.file_path, "w", encoding="utf-8") as writer:
                    json.dump(self.output, writer, cls=JSONEncoder)
            # Saves all parameters as object attributes to store in pickle
            elif output_file_type in [".p", ".pickle"]:
                with open(self.file_path, "wb") as writer:
                    # write to the file using binary mode
                    pickle.dump(self.output, writer, protocol=pickle.HIGHEST_PROTOCOL)
            # Saves all parameters as object attributes to store in xlsx
            elif output_file_type in [".xls", ".xlsx"]:
                try:
                    with pd.ExcelWriter(self.file_path) as writer:
                        for _, sheet_name, _ in self.column_sheet_mapping:
                            self.output[sheet_name].to_excel(
                                writer, index=True, sheet_name=sheet_name
                            )
                except ValueError as error:
                    if list(self.output.values())[0].shape[0] > 255:
                        raise ValueError(
                            "\n----------Write to Disk (2)----------\n"
                            "pandas.to_excel() is not capable to handle large data"
                            "with more than 255 columns. Please use other"
                            "file_extensions instead, e.g. 'json'."
                        ) from error
                    raise ValueError(error) from error
            elif output_file_type == ".csv":
                raise NotImplementedError

    def read_from_disk(self, read_filename, output_file_type=".json"):
        """
        Reads data from disk in various file formats and returns a dictionary
        or pandas dataframe.
        Args:
            self (object):
                The class instance calling the method.
            read_filename (str):
                The filename or filepath to read from.
            output_file_type (str, optional):
                The file format of the output file.
                Defaults to ".json".
        Returns:
            data (dict or pandas dataframe): The data read from the file.
        Raises:
            UserWarning: If an invalid output file type is specified.
        Example Usage:
            >>> pgmc = PowerGridManagerClass()
            >>> data = pgmc.read_from_disk('mydata', '.json')
        """
        # Read from disk
        if output_file_type in [".csv", ".xls", ".xlsx", ".json", ".p", ".pickle"]:
            read_file_path = os.path.join(read_filename + output_file_type)
        else:
            raise UserWarning(
                "Specify output file with .csv, .xls, .xlsx, .p or .json ending"
            )
        # read JSON
        if output_file_type == ".json":
            with open(read_file_path, "r", encoding="utf-8") as filepath:
                output_json = json.load(filepath)
                data = {key: pd.read_json(output_json[key]) for key in output_json}
        # read pickle
        elif output_file_type in [".p", ".pickle"]:
            with open(read_file_path, "rb") as reader:
                data = pickle.load(reader)
        # read xlsx
        elif output_file_type in [".xls", ".xlsx"]:
            data = {}
            for key, sheet_name, column_name in self.column_sheet_mapping:
                data_frame = pd.read_excel(
                    read_file_path, sheet_name=sheet_name, index_col=0
                )
                data_frame.columns = column_name
                data[sheet_name] = data_frame
        return data

    def read_check(self, df1, df2):
        """
        Compares two pandas dataframes to check if they are identical within a
        specified tolerance.
        Args:
            self (object): The class instance calling the method.
            df1 (pandas dataframe): The first dataframe to compare.
            df2 (pandas dataframe): The second dataframe to compare.
        Returns:
            None. The method prints a message indicating if the dataframes are
            identical within the tolerance or not.
        Example Usage:
            >>> pgmc = PowerGridManagerClass()
            >>> data1 = pgmc.read_from_disk('mydata1', '.csv')
            >>> data2 = pgmc.read_from_disk('mydata2', '.csv')
            >>> pgmc.read_check(data1, data2)
        """
        tolerance = 1e-6
        for _, key, _ in self.column_sheet_mapping:
            if (df1[key] - df2[key]).abs().max().max() > tolerance:
                return print(
                    f"\n----------Read Check (1)----------\n"
                    f"DataFrames df1 and df2 are outside the "
                    f"range of tolerance {tolerance}."
                )
            print((df1[key] - df2[key]).abs().max().max())
        return print(
            f"\n----------Read Check (1)----------\n"
            f"DataFrames df1 and df2 are identical each"
            f"within the tolerance value {tolerance}."
        )

    def get_grid_variables_to_log_or_store(self):
        """Returns a dictionary of grid component variables for both logging and storing purposes.
        This includes standard variables and additional variables specific to storage scenarios.
        Returns:
            dict: A dictionary where each key is a component name and
                  the value is a list of associated variables.
            get_variables_to_log
            get_variables_to_store
        """
        # Standard variables for grid components
        variables_config = {
            "res_load": ["p_mw", "q_mvar"],
            "sgen": ["p_mw", "q_mvar"],
            "res_bus": ["p_mw", "q_mvar", "vm_pu", "va_degree"],
            "res_ext_grid": ["p_mw", "q_mvar"],
            "res_line": ["pl_mw", "ql_mvar", "loading_percent"],
            "res_trafo": [
                "loading_percent",
                "p_hv_mw",
                "q_hv_mvar",
                "pl_mw",
                "ql_mvar",
            ],
            # Additional variables can be added here...
        }
        # Include additional storage-related variables if in a storage scenario
        if self.is_storage_scenario:
            storage_variables = {
                "storage": ["p_mw", "q_mvar", "soc_percent", "discharge_power_loss"]
            }
            variables_config.update(storage_variables)
        return variables_config

    def get_simulation_time_intervals(self):
        """
        Returns a list of time indices (default: in quarter-hour intervals) for
        the specified simulation time range.
            If a manual time range is selected, the indices are obtained from
            the user-specified start and end times.
            If a random time range is selected, the indices are randomly
            selected within the specified time range.
            If no time range is specified, the default time range is used.
        Notes (I):
        ------
        The simulation time range is determined based on the mode selected by
        the user arguments of configuration in get_simulation_time_intervals():
        either a manual selection of time intervals or a random a simulation
        time interval.
        """
        # Check which time mode is selected.
        if self.time_mode == "selected":
            # Manual select time (simulation steps)
            self.get_selected_sim_intervals()
        elif self.time_mode == "random":
            # Randomly select time range (simulation steps)
            self.get_random_sim_intervals()
        else:
            # Select default time range (self.sim_steps = range(self.max_iterations))
            self.get_default_sim_intervals()

    def get_selected_sim_intervals(self):
        """This function calculates the start time of the simulation episode in
        intervals, based on the selected start hour, start day, and interval
        resolution.
        It then determines the total number of intervals needed for the episode
        plus history, and creates a list of intervals that will be used for the
        simulation. The function returns the list of simulation intervals.
        """
        # Calculate the start time of the episode in intervals
        interval_per_hour = (
            60 // self.resolution
        )  # (each interval is a fixed length of time)
        start = (
            self.episode_start_min_interval
            + (self.episode_start_hour * interval_per_hour)
            + (self.episode_start_day * 24 * interval_per_hour)
        )
        # Determine the total number of intervals needed for the episode plus history
        num_intervals = self.episode_limit  #  + self.history + 1 # (with a margin of 1)
        # Create a list of intervals that will be used for the simulation
        self.base_sim_steps = range(start, start + num_intervals)
        highest_step, lowest_step = max(self.base_sim_steps), min(self.base_sim_steps)
        self.sim_steps = range(0, (highest_step - lowest_step) + 1)
        # define max number of steps
        self.max_steps = highest_step - lowest_step - 1
        self.print_simulation_timestamp_settings(lowest_step, highest_step)

    def get_random_sim_intervals(self):
        """This function resets the time stamp and retrieves one episode of
        data for PV histories, active load histories, and reactive load
        histories. It then sets the demand and PV.
        """
        # Reset the timestamp to a random start time for the episode
        self.episode_start_hour = self._select_random_start_hour()
        self.episode_start_day = self._select_random_start_day()
        self.episode_start_min_interval = self._select_random_start_interval()
        # Berechnung des Datums
        # start_date = datetime(year, 1, 1) + timedelta(days=start_day - 1, hours=start_hour)
        # Retrieve an episode's worth of data for PV, active load, and reactive load histories
        self.pv_histories = self.get_episode_pv_history()
        self.active_load_histories = self.get_episode_active_load_history()
        self.reactive_load_histories = self.get_episode_reactive_load_history()
        # Initialize simulation intervals based on the randomly selected start time
        # This is necessary as the starting day and time are randomly generated
        self.get_selected_sim_intervals()

    def get_default_sim_intervals(self):
        """This function returns a range object representing the simulation
        intervals for the default maximum number of iterations."""
        self.base_sim_steps = range(self.max_iterations)
        self.sim_steps = range(self.max_iterations)
        highest_step, lowest_step = max(self.base_sim_steps), min(self.base_sim_steps)
        # define max number of steps
        self.max_steps = highest_step - lowest_step
        self.print_simulation_timestamp_settings(lowest_step, highest_step)

    def print_simulation_timestamp_settings(self, lowest_step, highest_step):
        """Prints the simulation timestamp settings.
        Args:
            lowest_step (int): The lowest step in the simulation.
            highest_step (int): The highest step in the simulation.
        Prints:
            - The simulation start date
            - The simulation end date
            - The maximum number of simulation steps
            - The number of simulation days
        Example:
            s = Simulation()
            s.print_simulation_timestamp_settings(lowest_step=0, highest_step=96)
        Output:
            -------------Simulation Timestamp settings-------------
            Simulation start date:      2023-03-16 00:15:00
            Simulation end date:        2023-03-16 23:45:00
            Maximum simulation steps:   96-steps
            Number of Simulation Days:  1-days
            -------------------------------------------------------
        """
        self.first_date_simulation = self.time_data.loc[
            self.time_data["timestamps"] == lowest_step
        ].index[0]
        self.last_date_simulation = self.time_data.loc[
            self.time_data["timestamps"] == highest_step
        ].index[0]
        print("-------------Simulation Timestamp settings-------------")
        print(f"Simulation Start Date:      {self.first_date_simulation}")
        print(f"Simulation End date:        {self.last_date_simulation}")
        print(f"Maximum Simulation Steps:   {self.max_steps}-steps")
        print(f"Number of Simulation Days:  {self.max_steps//96}-days")
        print("-------------------------------------------------------")

    def _select_random_start_hour(self):
        """This function randomly selects a start hour for an episode from the
        range 0 to 23. It returns the selected start hour."""
        return np.random.choice(24)

    def _select_random_start_day(self):
        """This function randomly selects a start day (date) for an episode,
        based on the timestamp data in the simulation. It calculates the number
        of days between the first and last timestamp, and then selects a
        random day within that range that is at least episode_days away from
        the end of the range, where episode_days is the number of days required
        for the simulation episode.
        It returns the selected start day (date) for an episode.
        """
        time_data = self.time_data
        timestamp_days = (time_data.index[-1] - time_data.index[0]).days
        self.resolution = (time_data.index[1] - time_data.index[0]).seconds // 60
        assert self.resolution == self.resolution, (
            f"Data format error, expected resolution: {self.resolution} "
            f"but received time_delta: {self.resolution}. Verify data compatibility!"
        )
        episode_days = (
            self.episode_limit // (24 * (60 // self.resolution)) + 1
        )  # (with a margin of 1)
        return np.random.choice(timestamp_days - episode_days)

    def _select_random_start_interval(self):
        """This function randomly selects a start interval for an episode from
        the range of intervals that make up an hour, based on the time
        resolution of the simulation.
        By default:
            the time resolution is set to 15 minutes (i.e., 60 minutes divided
            into 4 intervals of 15 minutes each).
            It returns the selected start interval.
        """
        return np.random.choice(60 // self.resolution)

    def get_episode_pv_history(self):
        """The function returns the photovoltaic (PV) history for an episode
        based on the selected start hour, start day, and interval resolution.
        It returns the PV history as a numpy array."""
        episode_length = self.episode_limit
        history = self.history
        start = (
            self.episode_start_min_interval
            + (self.episode_start_hour * (60 // self.resolution))
            + (self.episode_start_day * 24 * (60 // self.resolution))
        )
        nr_intervals = episode_length + history + 1  # (with a margin of 1)
        episode_pv_history = self.pv_data[start : start + nr_intervals].values
        return episode_pv_history

    def get_episode_active_load_history(self):
        """This function returns the active power histories for all loads in
        an episode based on the selected start hour, start day, and interval
        resolution. It returns the data source and returns the active power history.
        """
        episode_length = self.episode_limit
        history = self.history
        start = (
            self.episode_start_min_interval
            + (self.episode_start_hour * (60 // self.resolution))
            + (self.episode_start_day * 24 * (60 // self.resolution))
        )
        nr_intervals = episode_length + history + 1  # (with a margin of 1)
        episode_load_active_power_history = self.load_active_power_data[
            start : start + nr_intervals
        ].values
        return episode_load_active_power_history

    def get_episode_reactive_load_history(self):
        """This function returns the reactive power histories for all loads in
        an episode based on the selected start hour, start day, and interval
        resolution. It returns the data source and returns the reactive power history.
        """
        episode_length = self.episode_limit
        history = self.history
        start = (
            self.episode_start_min_interval
            + (self.episode_start_hour * (60 // self.resolution))
            + (self.episode_start_day * 24 * (60 // self.resolution))
        )
        nr_intervals = episode_length + history + 1  # (with a margin of 1)
        episode_load_reactive_power_history = self.load_reactive_power_data[
            start : start + nr_intervals
        ].values
        return episode_load_reactive_power_history
