"""
PowerGridRuleBasedControlPP Module
----------------------------------
This module defines the `PowerGridRuleBasedControlPP` class, which is an extension of the
`PowerGridRuleBasedControl` class. The primary distinction lies in the implementation of the
`timeseries_ctrl` configuration parameter.
"""

# Importing necessary libraries
import os
import copy
import pandas as pd

# Importing modules from Pandapower library
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower import control
from pandapower import timeseries

# Importing modules from subdirectories
from .storage_controller import StorageController
from .pv_controller import PVController

# Importing test modules from subdirectories
from .pv_controller_test import PVControllerTest

# Importing parent module
from .base_powergrid_extended import BasePowerGridExtended


class PowerGridRuleBasedControlPP(BasePowerGridExtended):
    """
    A class that implements a rule-based control method for grid storage in a
    a power grid.
    Parameters
    ----------
    args : dict
        A dictionary of configuration settings for the control method.
    Attributes
    ----------
    power_grid : pandapowerNet
        An instance of the class pandapowerNet representing the power grid.
    args : dict
        A dictionary with configuration settings for the control method.
    Methods
    -------
    run_control():
        Runs the rule-based control method for the network storage.
    """

    def __init__(self, args):
        super().__init__(args)
        # reset the power grid
        self.net = copy.deepcopy(self.net)
        self.pv_controller = None
        self.storage_controller = None

    def run_main(self):
        """Runs the power grid simulation according to the selected control mode.
        Control modes:
        - manual: the simulation is run using manual control inputs.
        - control_module: the simulation is run using a control module.
        Raises:
        - NotImplementedError: if the selected control mode is not valid.
        Usage:
        >>> pge_rbc_pp = PowerGridRuleBasedControlPP()
        >>> pge_rbc_pp.args = {"control_modes": {"timeseries_ctrl": "control_module"}}
        >>> pge_rbc_pp.run()
        """
        # Control Strategies for Power Grids
        if self.args["control_modes"]["timeseries_ctrl"] == "control_module":
            self._run_pandapower_ts_sim()
        elif self.args["control_modes"]["timeseries_ctrl"] == "test_mode":
            self._run_test_der()
        elif self.args["control_modes"]["timeseries_ctrl"] == "manual":
            # self._run_manual()
            raise NotImplementedError("\n---self._run_manual---n")
        else:
            raise NotImplementedError(
                f"\n----------Control Strategies for Power Grids-----------\n"
                f"ctrl_mode: {self.timeseries_ctrl} variable is not valid\n"
                f"Please select a valid value from the following options:\n"
                f"(1) '{self.valid_modes['timeseries_ctrl'][0]}'\n"
                f"(2) '{self.valid_modes['timeseries_ctrl'][1]}'\n"
            )

    def _run_pandapower_ts_sim(self):
        """Time series simulation in Pandapower with control module for
        generation, storage and (loads)"""
        print("\n----------PandaPower Time Series Simulation-----------\n")
        self.simulation_mode = "pandapower"

        # Standard-Simulation ohne Monte Carlo
        # (1) Initialization
        self.pp_pre_processing_data_controllers()  # pp controller
        self.pp_pre_processing_init_log_variable_generic()  # Presetting of log-variables

        # (2) Start of Power Flow Time-Series Calculation ((quasi)-static analysis)
        timeseries.run_timeseries(self.net)  # for n_ts(15 min resolution)
        self.pp_post_processing_log_store_generic()

        # (3) Saving results
        self.store_logs()  # Storing total results simulation(*)
        self.write_to_disk()  # Write to disk(*)

    def _run_test_der(self):
        """Time series simulation in Pandapower with control module for
        generation, storage and (loads)"""
        print("\n----------PandaPower Test Simulation-----------\n")
        self.simulation_mode = "pandapower"
        # (1) Initialization
        # self.pp_pre_processing_data_controllers() # pp controller
        # # Test Class for PV-Module
        pv_controller_test = PVControllerTest(
            net=self.net,
            pid=0,
            pv_control_mode=self.pv_ctrl,
            regulation_standard=self.regulation_standard,
            timeseries_ctrl=self.timeseries_ctrl,
            inverter_ctrl_params=self.inverter_ctrl_params,
            output_data_path=self.output_test_path,
        )
        pv_controller_test.run_rule_based_pv_control_vde_tests(self.net)
        print("self.output_test_path:", self.output_test_path)
        from .storage_controller_test import StorageControllerTest

        # Test Class for Storage-Module
        storage_controller_test = StorageControllerTest(
            net=self.net,
            sid=0,
            storage_p_control_mode=self.storage_p_ctrl,
            storage_q_control_mode=self.storage_q_ctrl,
            regulation_standard=self.regulation_standard,
            timeseries_ctrl=self.timeseries_ctrl,
            inverter_ctrl_params=self.inverter_ctrl_params,
            output_data_path=self.output_test_path,  # self.output_test_data_path
            data_source=0,
            profile_name=0,
            resolution=self.resolution,
            inital_soc=self.soc_initial,
            scale_factor=self.scaling_storage,
            mcs_settings=self.mcs_settings,
            order=1,
            level=0,
        )
        storage_controller_test.run_rule_based_control_vde_tests(self.net)

    def pp_pre_processing_data_controllers(self):
        """
        Preprocessing of simulation data and controls for the panda power grid.
        This method sets up simulation time intervals, loads data sources for
        Households, PV systems, and Storage units, and applies constant control
        settings  and updates to the corresponding Pandapower elements for the specified
        time period dynamically.
        for households:
            using _apply_profiles_data_household_control() [ConstControl]
        for PV:
            using _apply_profiles_data_pv_control() [PVController with pid=pid]
            for MPV:
                if self.mpv_flag is True [PVController with pid=mpv_idx]
        for Storage:
            using _apply_profiles_data_storage_control() [StorageController with sid=sid].
        The updates are continuous over time, with the selected control algorithm
        and inverter determining the PV and storage controls for active and
        reactive power at each time step (using a quasi-static time series, QSTS).
        The data is organized as a time series, with each value associated with
        a specific time (default: quarter-hourly, but can be changed).
        This method sets up simulation time intervals, loads data sources for
        PV systems, households, and storage units, and applies constant control
        settings and updating to the corresponding pandapower elements for the
        specified time range.
        Parameters:
        -----------
        self : object
            An instance of the class ('PowerGridManagerClass', pgmc) that
            contains the pandapower powergrid and simulation data.
        Returns:
        --------
            The method updates the simulation data and the pandapower grid
            elements for each timestamp over the time series (ts).
        Notes:
            (I), (II), (III), and (IV) in the function's docstring are
            listed in sequential order.
        """
        # (I) Returns a time range for the predefined simulation time range.
        self.get_simulation_time_intervals()
        # (II) Using household control in the powergrid
        self._apply_profiles_data_household_control()
        # (III) Using pv control in the powergrid
        self._apply_profiles_data_pv_control()
        # Check if storage is being used in this scenario or specified by args
        if self.is_storage_scenario is True:
            # (IV) Using storage control in the powergrid
            self._apply_profiles_data_storage_control()

    def _log_variable(self, output_writer, variable, attribute, **kwargs):
        """Helper function to log a variable"""
        output_writer.log_variable(variable, attribute, **kwargs)

    def _create_dataframe(self, output_writer, np_results_key):
        """Helper function to create a DataFrame from np_results"""
        return pd.DataFrame(data=output_writer.np_results[np_results_key])

    def pp_pre_processing_init_log_variable_generic(self):
        """Generic function to initialize log variables"""
        variables_to_log = self.get_grid_variables_to_log_or_store()
        # Create an instance of OutputWriter and configure it
        output_writer = OutputWriter(
            net=self.net,
            time_steps=self.base_sim_steps,
            output_path=self.output_data_path,
            output_file_type=".xlsx",
        )
        # Create the directory for output data if it doesn't already exist
        os.makedirs(self.output_data_path, exist_ok=True)
        # base_log_variables:
        for main_element, attributes in variables_to_log.items():
            for sub_attribute in attributes:
                self._log_variable(output_writer, main_element, sub_attribute)
        # Save the OutputWriter instance as an attribute
        self.output_writer = output_writer

    def pp_post_processing_log_store_generic(self):
        """Generic function to store log variables"""
        # Retrieve the dictionary of variables to store
        variables_to_store = self.get_grid_variables_to_log_or_store()
        # Initialize an empty dictionary to log the dataframes
        log_variables = {}
        # Iterate through each main_element and its attributes in the dictionary
        for main_element, attributes in variables_to_store.items():
            # Iterate through each attribute in the current main_element
            for sub_attribute in attributes:
                # Construct the full variable name by combining main_element and sub_attribute
                var_name = f"{main_element}.{sub_attribute}"
                # Create a DataFrame for the current variable
                dataframe = self._create_dataframe(self.output_writer, var_name)
                # Store the DataFrame in the dictionary using the variable name as the key
                log_variables[var_name] = dataframe
        # Store the post-processed simulation results as log_variables to the class attribute
        self.log_variables = log_variables

    def _apply_profiles_data_household_control(self):
        """
        Applies household control to the pandapower network for the specified
        simulation steps.
        Parameters
        ----------
        self : object
            The pandapower network object.
        sim_steps : list or range object
            The time steps for which to apply the household control.
        Notes (II)
        ----------
        This function '_apply_profiles_data_household_control()' loads active and reactive power
        data sources for households at the specified simulation steps in the
        powergrid.
        A constant control `ConstControl` class (in the future 'LoadController'
        class) setting is applied for each household's load element in the grid
        using the corresponding power data source.
        The control is applied to both the active power (p_mw) and reactive power
        (q_mvar) variables of the load element (households) in the powergrid,
        indexed by simulation time steps (rows).
        The data source is defined using a DFData object, and the control is
        applied using the ConstControl class from the pandapower.control module.
        The profile name for the load element is set to the index of the load
        element in the network.
        """
        # Active Load power block
        # Load active power data source for household units with self.base_sim_steps
        self.profiles[("load", "p_mw")] = copy.deepcopy(
            self.load_active_power_data.iloc[self.base_sim_steps, :].reset_index(
                drop=True
            )
        )
        # Create a data source object from the loaded active power data
        load_active_power_ds = DFData(self.profiles[("load", "p_mw")])
        # For each household unit, apply the ConstControl for active power
        control.ConstControl(
            self.net,
            element="load",
            element_index=self.net.load.index,
            variable="p_mw",
            data_source=load_active_power_ds,
            profile_name=self.net.load.index,
        )
        # Reactive Load power block
        # Load reactive power data source for household units with self.base_sim_steps
        self.profiles[("load", "q_mvar")] = copy.deepcopy(
            self.load_reactive_power_data.iloc[self.base_sim_steps, :].reset_index(
                drop=True
            )
        )
        # Create a data source object from the loaded active power data
        load_reactive_power_ds = DFData(self.profiles[("load", "q_mvar")])
        # For each household unit, apply the ConstControl for reactive power
        control.ConstControl(
            self.net,
            element="load",
            element_index=self.net.load.index,
            variable="q_mvar",
            data_source=load_reactive_power_ds,
            profile_name=self.net.load.index,
        )

    def _apply_profiles_data_pv_control(self):
        """
        Applies control settings for photovoltaic (PV) systems to the pandapower
        network for the entire simulation time range.
        Parameters
        ----------
        self : object
            The pandapower network object.
        sim_steps : list or range object
            The time steps for which to apply the PV control.
        Notes (III)
        -----------
        This function loads PV power data sources for the specified simulation
        steps.
        A control setting is applied to each PV system element in the network
        using the corresponding power data source.
        The control setting depends on the selected algorithm, using the
        user-implemented "PVController" class or the default "ConstControl"
        class from the "pandapower.control" module to apply the control setting
        for each PV system.
        The control is applied to the active power variable (p_mw) of the sgen
        (synchronous generator) and the apparent power (sn_mva) and reactive
        power (q_mvar) of the PV inverters are determined.
        The data source is defined using a DFData object, and the profile name
        for the sgen element is set to the index of the sgen element (cols).
        It is important:
        The PV reactive power is jointly calculated with the active PV power
        in the StorageController class.
        """
        # PV block
        # Load active power data source for pv units for specified simulation range
        self.profiles[("sgen", "p_mw")] = copy.deepcopy(
            self.pv_data.iloc[self.base_sim_steps, :].reset_index(drop=True)
        )
        # Load reactive power data source for pv units for specified simulation range
        self.profiles[("sgen", "q_mvar")] = (
            copy.deepcopy(
                self.pv_data.iloc[self.base_sim_steps, :].reset_index(drop=True)
            )
            * 0
        )
        pv_active_power_df = self.profiles[("sgen", "p_mw")]
        # Create a data source object from the loaded data
        pv_active_power_ds = DFData(pv_active_power_df)
        # For each pv unit, apply the PVController using data source
        # for pid, pv_idx in enumerate(self.net.sgen.index):
        for pid, pv_idx in enumerate(self.pv_range_index):
            # Instead of using range(len(self.net.storage.index)), the
            # enumerate function stores both the index and the value of
            # each element in self.net.storage.index.
            # self.net.sgen.index[i] == pv_idx
            self.pv_controller = PVController(
                net=self.net,
                pid=pid,
                pv_control_mode=self.pv_ctrl,
                regulation_standard=self.regulation_standard,
                timeseries_ctrl=self.timeseries_ctrl,
                inverter_ctrl_params=self.inverter_ctrl_params,
                output_data_path=self.output_data_path,
                resolution=self.resolution,
                data_source=pv_active_power_ds,
                profile_name=pv_idx,
                scale_factor=self.scaling_pv,
                order=0,
                level=pv_idx,
            )
        if self.mpv_flag:
            print(self.mpv_flag)
            # Load MPV data (Mini Photovoltaic energy generation)
            self.mpv_data = self._load_mpv_data()
            self.profiles[("mpv_sgen", "p_mw")] = copy.deepcopy(
                self.mpv_data.iloc[self.base_sim_steps, :].reset_index(drop=True)
            )
            self.profiles[("mpv_sgen", "q_mvar")] = (
                copy.deepcopy(
                    self.mpv_data.iloc[self.base_sim_steps, :].reset_index(drop=True)
                )
                * 0
            )
            mpv_active_power_df = self.profiles[("mpv_sgen", "p_mw")]
            mpv_active_power_ds = DFData(mpv_active_power_df)
            for mpvid, mpv_idx in enumerate(self.mpv_range_index):
                print(f"mpvid: {mpvid}, mpv_idx: {mpv_idx}")
                self.mpv_controller = PVController(
                    net=self.net,
                    pid=mpv_idx,
                    pv_control_mode=self.pv_ctrl,
                    regulation_standard=self.regulation_standard,
                    timeseries_ctrl=self.timeseries_ctrl,
                    inverter_ctrl_params=self.inverter_ctrl_params,
                    output_data_path=self.output_data_path,
                    resolution=self.resolution,
                    data_source=mpv_active_power_ds,
                    profile_name=pv_idx,
                    scale_factor=self.scaling_pv,
                    order=0,
                    level=pv_idx,
                )

    def _apply_profiles_data_storage_control(self):
        """
        Applies the specified control settings for storage units for the entire
        simulation time range. The control settings depend on the selected
        algorithm and include control for charging and discharging power of the
        storage unit.
        Parameters:
        -----------
        sim_steps : list
            A list of simulation time steps for which the control settings need
            to be applied.
        Notes: (IV)
        -----------
        This function loads the datasource and control for the storage units.
        If the scenario specified is in the `sb_storage_scenarios` list or if
        the `usage` flag in the `storage` dictionary is set to `True`,
        the control settings are applied to the corresponding storage elements
        for the specified time range.
        The updates occur continuously over time, depending on the control
        algorithm and the storage controller selected.
        """
        # Set the initial power to 0 and apply storage control
        self.net.storage["p_mw"] = 0.0
        # Storage block
        # Load active data source for storage units for specified simulation range
        self.profiles[("storage", "p_mw")] = copy.deepcopy(
            self.storage_active_power.iloc[self.base_sim_steps, :].reset_index(
                drop=True
            )
        )
        # Load reactive data source for storage units for specified simulation range
        self.profiles[("storage", "q_mvar")] = copy.deepcopy(
            self.storage_reactive_power.iloc[self.base_sim_steps, :].reset_index(
                drop=True
            )
        )
        # Create a data source object from the loaded data
        storage_active_power_ds = DFData(self.profiles[("storage", "p_mw")])
        # For each storage unit, apply the StorageController using data source
        for sid, storage_idx in enumerate(self.net.storage.index):
            # Instead of using range(len(self.net.storage.index)), the
            # enumerate function stores both the index and the value of
            # each element in self.net.storage.index.
            # self.net.storage.index[i] == storage_idx
            self.storage_controller = StorageController(
                net=self.net,
                sid=sid,
                storage_p_control_mode=self.storage_p_ctrl,
                storage_q_control_mode=self.storage_q_ctrl,
                regulation_standard=self.regulation_standard,
                timeseries_ctrl=self.timeseries_ctrl,
                inverter_ctrl_params=self.inverter_ctrl_params,
                output_data_path=self.output_data_path,
                data_source=storage_active_power_ds,
                profile_name=storage_idx,
                resolution=self.resolution,
                inital_soc=self.soc_initial,
                scale_factor=self.scaling_storage,
                mcs_settings=self.mcs_settings,
                order=1,
                level=storage_idx,
            )
