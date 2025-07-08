"""
BasePowerGridCore Module
-------------------------
The `BasePowerGridCore` module defines foundational classes for power grid control strategies.
It serves as a base for loading configuration parameters, grid topologies, and raw profile data
for power grid elements such as loads, photovoltaic (PV) generation, household batteries, and
electricity price time series.

- **BasePowerGridCore**: Provides core functionality for handling grid configurations,
  profiles, and topology management.
- **BasePowerGridExtended**: Extends `BasePowerGridCore` by adding attributes and methods for
  implementing specific control strategies like `PowerGridRuleBasedControlPP` and
  `PowerGridRuleBasedControl`.
"""

# Importing necessary libraries
import sys
import os
import datetime
import copy
from collections import namedtuple
import numpy as np
import pandas as pd
import networkx as nx

# Importing modules from Pandapower and SimBench libraries
import pandapower as pp
import simbench
from pandapower.topology.create_graph import create_nxgraph

# Importing configuration datatype library
import yaml

# Importing modules from subdirectories (Load a toy grid used for testing purposes)
from .helper_powergrid_customised_grid import toy_grid


class BasePowerGridCore:
    """
    Base class for power grids.
    """

    def __del__(self):
        """Destroys the instance of the object"""

    def __init__(self, kwargs):
        """
        Initializes the BasePowerGridCore object with the given keyword arguments.
        Args:
            kwargs: Dictionary containing the keyword arguments.
        """
        # Load the main configuration settings
        self.config_file = self.load_configuration_settings(kwargs)
        # Save configuration settings
        self.args = self.config_file

        # Set time-related parameters
        self._set_time_parameters()
        # Set Monte Carlo-related parameters
        self._set_monte_carlo_runs_parameters()
        # Set Mini Photovoltaic (MPV) settings parameters
        self._set_mpv_parameters()
        # Set input and output paths
        self._set_input_output_paths()
        # Set simbench episode settings
        self._set_simbench_episode_settings()
        # Set simulation controls settings
        self._set_simulation_controls_settings()
        # Set plot settings
        self._set_eval_plot_settings()

        # Load data and model
        self.load_data_and_model()
        # Update voltage in inverter control parameters
        self.update_voltage_inverter_control_parameters()
        # Calcculate and display reactive power boundaries
        self.calculate_and_display_reactive_power_boundaries()
        # Resets the power grid to its original state
        self.reset_to_initial_state()
        # Save the initial/refined state of the power grid to an Excel file
        net_json_filename = os.path.join(self.output_data_path, "net1.json")
        pp.to_json(net=self.base_powergrid, filename=net_json_filename)
        # Save sim info data
        self._get_penetration_rates()

    @classmethod
    def timer(cls, func):
        """
        Decorator function that measures the elapsed time of a given function.
        Args:
            cls: Class on which the method is called.
            self: Instance of the class on which the method is called.
            func: Function to be timed.
        Returns:
            Wrapper function that measures the elapsed time of the given function.
        """

        def wrapper(self, *args, **kwargs):
            """
            Inner function that measures the elapsed time of the decorated function.
            Args:
                self: Instance of the class on which the method is called.
                *args: Positional arguments to be passed to the decorated function.
                **kwargs: Keyword arguments to be passed to the decorated function.
            Returns:
                Return value of the decorated function.
            """
            before = datetime.datetime.now().timestamp()
            val = func(self, *args, **kwargs)
            elapsed_time = (datetime.datetime.now().timestamp() - before) * 1000.0
            self.runtime[func.__name__ + "_elapsed_time"] = elapsed_time
            return val

        return wrapper

    @staticmethod
    def convert(dictionary):
        """
        Converts a Python dictionary to a named tuple with the keys of the dictionary
        as field names.
        Args:
            dictionary: Python dictionary.
        Returns:
            Named tuple with the same keys as the input dictionary.
        """
        return namedtuple(typename="GenericDict", field_names=dictionary.keys())(
            **dictionary
        )

    @staticmethod
    def convert_watt_to_megawatt(watt_value):
        """Converts a value from watts to megawatts.
        Args: watt_value: The value in watts to be converted.
        Returns: The value in megawatts.
        """
        megawatt_value = watt_value / 1e6
        return megawatt_value

    @staticmethod
    def load_configuration_file(cfg_filename):
        """
        Load a configuration file.
        Args:
            cfg_filename (str): Path to the configuration file.
        Returns:
            dict: Dictionary containing the configuration settings.
        """
        with open(cfg_filename, encoding="utf-8") as stream:
            try:
                loaded_cfg = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print("Error in configuration file:", exc, stream.name)
                sys.exit()
        return loaded_cfg

    def _set_time_parameters(self):
        """Set the time-related parameters maxTime and deltaTime from the object's
        arguments."""
        # Variables used as input parameters in Storage and PV Control Modes
        # Set the simulation time resolution (default: 15 [min])
        self.resolution = self.args["sim"]["resolution"]
        # self.deltaTime = self.args['deltaTime']
        # Based on the time resolution, calc the number of time steps per day (default: 96 [15min])
        self.timesteps_per_day = int(24 * 60 / self.resolution)
        # Establish the maximum iteration count for a simulation of this environment.
        self.max_iterations = self.args["sim"]["max_iterations"]
        # self.maxTime = self.args['maxTime']

    def _set_mpv_parameters(self):
        """Sets the parameters for the Mini Photovoltaic (MPV) system.
        This method configures various settings for the MPV, including:
        - mpv_flag: A flag to indicate MPV usage.
        - mpv_benchmark: The benchmark parameter for MPV.
        - mpv_concentration_rate_percent: The concentration rate in percentage.
        - mpv_inverter_apparent_power_watt: The first gamma1 parameter for MPV adjustments.
        - mpv_solar_cell_capacity_watt: The second gamma2 parameter for MPV adjustments.
        """
        self.mpv_settings = self.args["mpv_settings"]
        self.mpv_flag = self.args["mpv_settings"]["mpv_flag"]
        self.mpv_benchmark = self.args["mpv_settings"]["mpv_benchmark"]
        self.mpv_scaling = self.args["mpv_settings"]["mpv_scaling"]
        self.mpv_concentration_rate_percent = self.args["mpv_settings"][
            "mpv_concentration_rate_percent"
        ]
        self.mpv_inverter_apparent_power_mva = self.convert_watt_to_megawatt(
            self.args["mpv_settings"]["mpv_inverter_apparent_power_watt"]
        )
        self.mpv_solar_cell_capacity_mw = self.convert_watt_to_megawatt(
            self.args["mpv_settings"]["mpv_solar_cell_capacity_watt"]
        )

    def _set_monte_carlo_runs_parameters(self):
        """Set the Monte Carlo-related parameters numMonteCarloRuns and random
        seed for each type of modelling error defined in the configuration and
        print the seed values."""
        self.mcs_settings = self.args["mcs_settings"]
        self.log_variables_mcs_data = []  # raw_log_mcs_data
        self.flag_monte_carlo = self.args["mcs_settings"]["flag_monte_carlo"]
        self.num_monte_carlo_runs = self.args["mcs_settings"]["num_monte_carlo_runs"]
        self.loc = self.args["mcs_settings"]["loc"]
        self.error_std_dev = self.args["mcs_settings"]["error_std_dev"]
        self.seed_value = self.args["mcs_settings"]["seed_value"]
        self.add_noise = self.args["mcs_settings"]["add_noise"]
        # Set the seed for the random generator (e.g. to the value 42)
        np.random.seed(self.seed_value)
        print("--------------------Random Seed Settings--------------------")
        print(f"{'flag_monte_carlo:':<35}{self.flag_monte_carlo}")
        print(f"{'num_monte_carlo_runs:':<35}{self.num_monte_carlo_runs}")
        print(f"{'loc:':<35}{self.loc}")
        print(f"{'error_std_dev:':<35}{self.error_std_dev}")
        print(f"{'seed_value:':<35}{self.seed_value}")
        print(f"{'add_noise:':<35}{self.add_noise}")

    def _set_input_output_paths(self):
        """Set the paths and file types for input and output data in the object's
        arguments and store them in corresponding attributes."""
        # Set the paths and file types for input and output data.
        self.run_main_base_path = self.args["cfg_settings"]["run_main_base_path"]
        self.helper_path = self.args["cfg_settings"]["helper_path"]
        self.input_data_path = self.args["cfg_settings"]["input_data_path"]
        self.output_data_path = self.args["cfg_settings"]["output_data_path"]
        self.output_test_path = self.args["cfg_settings"][
            "output_test_path"
        ]  # output_test_data_path
        self.sequence_id_path = self.args["cfg_settings"]["sequence_id_path"]

        self.input_data_type = self.args["cfg_settings"]["input_data_type"]
        self.output_file_type = self.args["cfg_settings"][
            "output_file_type"
        ]  # output_file_type

    def _set_simbench_episode_settings(self):
        """Set the simbench mode, simulation code, and simulation parameters for
        an episode."""
        # Set the simbench mode and code for the simulation.
        self.benchmark = self.args["rawdata"]["benchmark"]
        self.sb_code = self.args["rawdata"]["sb_code"]
        self.scenario = self.args["rawdata"]["scenario"]
        # Set the sb_code and scenario for the simulation.
        self.valid_sb_code = self.args["rawdata"]["valid_sb_code"]
        self.valid_sb_base_codes = self.args["rawdata"]["valid_sb_base_codes"]
        self.valid_sb_scenario = self.args["rawdata"]["valid_sb_scenario"]
        self.valid_sb_scenario_storage = self.args["rawdata"][
            "valid_sb_scenario_storage"
        ]
        # Set scenario name
        self._valid_and_set_scenario_name()

        self.soc_initial = self.args["rawdata"]["soc_initial"]
        # Set the simulation parameters for the episode.
        self.time_mode = self.args["sim"]["time_mode"]
        self.episode_start_hour = self.args["sim"]["episode_start_hour"]
        self.episode_start_day = self.args["sim"]["episode_start_day"]
        self.episode_start_min_interval = self.args["sim"]["episode_start_min_interval"]
        self.episode_limit = self.args["sim"]["episode_limit"]

    def _set_simulation_controls_settings(self):
        """Set the controller for specified elements and variables, and the control
        modes for timeseries, storage, and PV."""
        # Controller for a specified elements and variable
        # Set Timeseries Control Mode
        self.timeseries_ctrl = self.args["control_modes"]["timeseries_ctrl"]
        self.scaling_load = self.args["rawdata"]["scaling"]["load"]
        # Set Storage Control Mode
        self.storage_p_ctrl = self.args["control_modes"]["storage_p_ctrl"]
        self.storage_q_ctrl = self.args["control_modes"]["storage_q_ctrl"]
        self.scaling_storage = self.args["rawdata"]["scaling"]["storage"]
        # Set Wind Control Mode
        # self.scaling_wind = self.args["rawdata"]["scaling"]["wind"]
        # Set PV Control Mode
        self.pv_ctrl = self.args["control_modes"]["pv_ctrl"]
        self.scaling_pv = self.args["rawdata"]["scaling"]["pv"]

    def _set_eval_plot_settings(self):
        """Set the controller for specified elements and variables, and the control
        modes for timeseries, storage, and PV."""
        # Controller for a specified elements and variable
        # Set Timeseries Control Mode
        self.cfg_default_plot_path = self.args["cfg_settings"]["cfg_default_plot_path"]
        self.cfg_user_plot_path = self.args["cfg_settings"]["cfg_user_plot_path"]
        self.yaml_path = self.args["cfg_settings"]["yaml_path"]

    def _integrate_mpv_sgens_into_load_buses_for_MPVs(self):
        """Creates mini-photovoltaic (MPV) synchronous generators (MPVsgens) in the
        power grid based on the concentration rate of MPVs. This means the concentration
        rate percent of the MPVs is in relation to the number of MPVs divided by the
        number of load buses. 100 % means that an MPV has been added to all loads.
        Args:
            self.base_powergrid (pandapowerNet): The pandapower network where sgens will be added.
            self.mpv_concentration_rate_percent: Define the concentration rate of MPVs in percent (e.g., 100%).
        Returns:
            self.base_powergrid (pandapowerNet): The function modifies the grid directly and returns it.
        """
        try:
            pv_stop_index = self.pv_range_index.stop
            # Retrieve the list of load buses from the grid
            load_buses = self.base_powergrid.load.bus
            # Calculate the total number of load buses in the grid
            total_loads = len(self.base_powergrid.load)
            # Determine the number of load buses influenced by the mpv penetration rate
            num_mpv_influenced_load_buses = int(
                (self.mpv_concentration_rate_percent / 100) * total_loads
            )
            # Retrieve the list of static generator buses from the grid
            sgen_buses = self.base_powergrid.sgen.bus
            # Define the method for selecting bus IDs
            # Options: "random" for random selection, "sequential" for sequential selection
            selection_method = "random"
            # Output configuration and calculated parameters
            print("---------------------- Integrate MPV -----------------------")
            print(f"{'Selection Method of MPVs:':<35}{selection_method:>6s}")
            print(
                f"{'mpv_concentration_rate_percent:':<35}{self.mpv_concentration_rate_percent:>4.2f}%"
            )
            print(
                f"{'num_mpv_influenced_load_buses:':<35}{total_loads:>2d} total loads of {num_mpv_influenced_load_buses:>2d} mpvsgens"
            )
            print(
                f"{'mpv_inverter_apparent_power_mw:':<35}{self.mpv_inverter_apparent_power_mva:>2.6f} MW"
            )
            print(
                f"{'mpv_solar_cell_capacity_mw:':<35}{self.mpv_solar_cell_capacity_mw:>2.6f} MW"
            )
            print(f"{'total_loads:':<35}{total_loads:>2d}")
            for i in range(num_mpv_influenced_load_buses):
                if selection_method == "random":
                    # Using the random module to select a load bus ID and sgen bus ID
                    selected_index = np.random.choice(load_buses.index)
                    selected_load_bus_id = load_buses[selected_index]
                    selected_sgen_bus_id = np.random.choice(sgen_buses)
                elif selection_method == "sequential":
                    # Using the modulo operator for the case where there are fewer sgens than loads
                    selected_index = i % len(load_buses)
                    selected_load_bus_id = load_buses.iloc[selected_index]
                    selected_sgen_bus_id = sgen_buses[i % len(sgen_buses)]
                bus_condition = self.base_powergrid.sgen["bus"] == selected_load_bus_id
                if selected_load_bus_id in self.base_powergrid.sgen["bus"].values:
                    name_condition = ~self.base_powergrid.sgen["name"].str.contains(
                        "SGenMPV"
                    )
                    sgen_at_bus = self.base_powergrid.sgen[
                        bus_condition & name_condition
                    ]
                else:
                    sgen_at_bus = self.base_powergrid.sgen[bus_condition]
                # Entfernen Sie das ausgewählte Element aus der Liste
                load_buses = load_buses.drop(selected_index)
                sn_mva = self.mpv_inverter_apparent_power_mva
                p_mw = sn_mva * 0.99
                q_mvar = 0.0
                max_p_mw = self.mpv_inverter_apparent_power_mva
                min_p_mw = sgen_at_bus["min_p_mw"].iloc[0]
                sgen_type = "MPV"
                voltLvl = sgen_at_bus["voltLvl"].iloc[0]
                profile = sgen_at_bus["profile"].iloc[0]
                phys_type = sgen_at_bus["phys_type"].iloc[0]
                subnet = sgen_at_bus["subnet"].iloc[0]
                sgen_bes_name = (
                    f"PV{selected_sgen_bus_id}-Load{selected_load_bus_id} SGenMPV {i}"
                )
                pp.create_sgen(
                    net=self.base_powergrid,
                    bus=selected_load_bus_id,
                    p_mw=p_mw,
                    q_mvar=q_mvar,
                    sn_mva=sn_mva,
                    max_p_mw=max_p_mw,
                    min_p_mw=min_p_mw,
                    name=sgen_bes_name,
                    type=sgen_type,
                    voltLvl=voltLvl,
                    profile=profile,
                    subnet=subnet,
                    phys_type=phys_type,
                )
                # Formatierung der Ausgabe
                formatted_part1 = (
                    f" SGenMPV {i} added to grid in Bus Id {selected_load_bus_id}."
                )
                formatted_part2 = f"{total_loads}/{i + 1:02}"
                formatted_line = f"{formatted_part1:<40} Total Bus: {formatted_part2}"
                print(formatted_line)
            # Extract numbers from entries with "SGenMPV" in their names
            name_condition = self.base_powergrid.sgen.name.str.contains("SGenMPV")
            extract_sgen_mpv_num = (
                self.base_powergrid.sgen.name[name_condition]
                .str.extract("(\d+)$")
                .astype(int)
            )
            # Count entries containing "SGenMPV":
            self.mpvsgen_num = self.base_powergrid.sgen.name.str.contains(
                "SGenMPV"
            ).sum()
            # Determine the highest number among these entries
            self.mpvsgen_min_idx = extract_sgen_mpv_num.index.min() - 1
            self.mpvsgen_max_idx = extract_sgen_mpv_num.index.max()
            # Compare the count of entries to the highest number + 1
            self.mpvsgen_idx_values = extract_sgen_mpv_num.index
            self.mpv_range_index = self.base_powergrid.sgen.index
            mpv_stop_index = self.mpv_range_index.stop
            mpv_step_index = self.mpv_range_index.step
            self.mpv_range_index = pd.RangeIndex(
                start=pv_stop_index, stop=mpv_stop_index, step=mpv_step_index
            )
            print(f"{'mpvsgen_index_values:':<25}{self.mpvsgen_idx_values.tolist()}")
            if (self.mpvsgen_num + self.mpvsgen_min_idx) == self.mpvsgen_max_idx:
                print("An MPV has been added to each load in a full set")
            else:
                print("An MPV has been added to each load in a subset")
            # return mpv_sgens, net
            print(f"{'-'*60}")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

    def _integrate_sgens_into_storage_buses_for_PVBES(self):
        """Integrates additional photovoltaic (PV) systems or static generators (sgens)
        into the power network, focusing on storage buses that do not have PV systems.
        Args:
            self.base_powergrid (pandapowerNet): The pandapower network where sgens will be added.
        Returns:
            self.base_powergrid (pandapowerNet): The function modifies the grid directly and returns it.
        """
        try:
            storage_buses = self.base_powergrid.storage.bus
            sgen_buses = self.base_powergrid.sgen.bus
            num_sgen = len(self.base_powergrid.sgen)
            self.pv_range_index = self.base_powergrid.sgen.index
            print("--------------------- Integrate PV-BES ---------------------")
            print(f"{'num_sgen:':<35}{num_sgen}")
            for i in range(len(storage_buses)):
                storage_bus_id = storage_buses[i]
                sgen_bus_id = sgen_buses[i]
                sgen_at_bus = self.base_powergrid.sgen[
                    self.base_powergrid.sgen["bus"] == sgen_bus_id
                ]
                p_mw = sgen_at_bus["p_mw"].iloc[0]
                q_mvar = sgen_at_bus["q_mvar"].iloc[0]
                sn_mva = sgen_at_bus["sn_mva"].iloc[0]
                sgen_type = sgen_at_bus["type"].iloc[0]
                voltLvl = sgen_at_bus["voltLvl"].iloc[0]
                profile = sgen_at_bus["profile"].iloc[0]
                phys_type = sgen_at_bus["phys_type"].iloc[0]
                subnet = sgen_at_bus["subnet"].iloc[0]
                sgen_bes_name = (
                    f"{i+1}: PV{sgen_bus_id}-BES{storage_bus_id} SGen {i+num_sgen}"
                )
                pp.create_sgen(
                    net=self.base_powergrid,
                    bus=storage_bus_id,
                    p_mw=p_mw,
                    q_mvar=q_mvar,
                    sn_mva=sn_mva,
                    name=sgen_bes_name,
                    type=sgen_type,
                    voltLvl=voltLvl,
                    profile=profile,
                    subnet=subnet,
                    phys_type=phys_type,
                )
                print(f" added sgen no {'i=':<1}{sgen_bes_name}")
            print(f"{'num_sgen:':<35}{len(self.base_powergrid.sgen)}")

        except Exception as e:
            print(f"An error occurred: {e}")

    def load_data_and_model(self):
        """Load the power grid model, power system data, and price timeseries."""
        # Load the model of the power grid.
        self.base_powergrid = self._load_powergrid_network()

        # Integrate PV into BES for PVBES
        self.pv_range_index = self.base_powergrid.sgen.index
        self._integrate_sgens_into_storage_buses_for_PVBES()
        self.pv_range_index = self.base_powergrid.sgen.index
        print(f"self.mpv_flag_ {self.mpv_flag}")
        if self.mpv_flag:
            print(f"self.mpv_flag_ {self.mpv_flag}")
            self._integrate_mpv_sgens_into_load_buses_for_MPVs()
            print(f"self.mpv_flag_ {self.mpv_flag}")

        # Load power system data
        # Get timeseries with absolute values
        self.base_profiles = simbench.get_absolute_values(
            self.base_powergrid, profiles_instead_of_study_cases=True
        )
        # Load PV data (Photovoltaic energy generation)
        self.pv_data = self._load_pv_data()
        if self.mpv_flag:
            # Load MPV data (Mini Photovoltaic energy generation)
            self.mpv_data = self._load_mpv_data()
        # Load Household data
        self.load_active_power_data = self._load_loads_active_power_data()
        self.load_reactive_power_data = self._load_loads_reactive_power_data()

        # Load EV data (Electric Vehicle charging energy)
        # self.ev_data = self._load_ev_data()
        # Load HP data (Heat Pump energy consumption)
        # self.heat_pump_data = self._load_heat_pump_data()
        # Load time data
        self.time_data = self._load_time_data()
        # Load price timeseries
        # self.price_data = self._load_price_data()
        # Additional_constraints (storage usage) to simulation timestep
        self._load_and_apply_storage_constraints()
        # Retrieve zone data
        self.zone_data = self.get_zone_data()

    def update_voltage_inverter_control_parameters(self):
        """Retrieve the nominal grid voltage from the base power grid and update
        the inverter control parameters accordingly."""
        # Retrieve the nominal grid voltage value (in per unit)
        v_nom_net_value = self.base_powergrid.ext_grid.vm_pu.values[0]
        # Update the nominal grid voltage in inverter control parameters
        self.inverter_ctrl_params[self.regulation_standard][self.standard_mode][
            "v_nom_net"
        ] = v_nom_net_value

    def calculate_and_display_reactive_power_boundaries(self):
        """Calculate and display the reactive power boundaries based on the
        maximum active and apparent power from PV installations within the
        distribution powergrid and the maximum active power from the timeseries
        for PV and load.
        The function calculates the power factor, PV-to-load ratio, and prints
        these values for information and further analysis."""
        # Maximum active power of PV systems within the distribution grid
        p_max = self.base_powergrid.sgen.p_mw.max()
        # Maximum apparent power of PV systems inverter within the distribution grid
        s_max = self.base_powergrid.sgen.sn_mva.max()
        # Power factor
        power_factor = s_max / p_max
        # Maximum active power from the time series
        pv_active_max = self.pv_data.values.max()
        # Maximum active power from the load time series
        load_active_max = self.load_active_power_data.values.max()
        # PV maximum to load maximum ratio
        pv_load_ratio = pv_active_max / load_active_max
        # Print analysis data
        print("------------Analysis of Reactive Power Boundaries-----------")
        print(f"{'Rated power (s_max):':<35}{s_max:>4.2f} MVA")
        print(f"{'Power Factor (s_max/p_max):':<35}{power_factor:>4.2f} %")
        print(f"{'Maximum Active Power of PV:':<35}{pv_active_max:>4.2f} MW")
        print(f"{'Maximum Active Power of Load:':<35}{load_active_max:>4.2f} MW")
        print(
            f"{'PV Load Power Ratio (PV Max / Load Max):':<35}{pv_load_ratio:>4.2f} MW"
        )
        # Maximum MPV active power from the time series
        # mpv_active_max = self.mpv_data.values.max()
        # print("------------MPV Capacity -----------\n"
        #       f"Penetration of MPV Systems: {s_max} MVA\n"
        #       f"Concentration Rate Percent of MPV Systems: {s_max} MVA\n")

    def reset_to_initial_state(self):
        """Reset the power grid to its original state, update the scenario name
        and zone data, and update the configuration arguments to match the original
        state of the power grid."""
        # Reset power grid to original state
        self.net = copy.deepcopy(self.base_powergrid)
        # Update configuration arguments
        self.args = self.config_file

    def load_configuration_settings(self, args):
        """Load configuration settings from a YAML file, merge configurations,
        process paths, set control modes, load inverter control settings,
        create a results folder, and write the updated configuration to disk.
        Args:
            args (dict): Dictionary containing the keyword arguments.
        Returns:
            dict: Dictionary containing the updated configuration settings.
        Raises:
            ValueError: If an invalid control mode is provided.
        """
        # Print KIT-IAI-logo
        self.print_logo_iai_kit()
        # Load arguments from the YAML file
        main_cfg = self.load_arguments_from_yaml_configuration(args)
        # Update master arguments
        self.update_arguments_from_master_script(main_cfg, args)
        # Set control modes
        self.set_control_modes(main_cfg, args)
        # Load inverter control settings
        self.load_inverter_control_settings(main_cfg, args)
        # Save the updated configuration
        self.save_updated_configuration(main_cfg)
        self.main_cfg = main_cfg
        # Check valid simulation settings in config file
        self._check_config_settings()
        return self.main_cfg

    def load_arguments_from_yaml_configuration(self, args):
        """Merge default and user configurations, and process script path arguments.
        Args:
            args (dict): Dictionary containing the keyword arguments.
        Returns:
            dict: Dictionary containing the merged configuration settings and
            updated paths.
        """
        # Block-1: Load and merge configurations
        cfg_default = self.load_configuration_file(args.cfg_default_path)
        cfg_user = self.load_configuration_file(args.cfg_user_path)

        # Merge configurations
        self._merge_dicts(cfg_default, cfg_user)
        main_cfg = cfg_user  # Merged configuration becomes main configuration

        # Block-2: Process path arguments
        # input_dataset_path = main_cfg["cfg_settings"]["input_data_path"]
        output_dataset_path = main_cfg["cfg_settings"]["output_data_path"]
        config_folder_path = os.path.dirname(args.cfg_default_path)

        # Block-3: Generate suffixes for filenames
        suffix_episode_limit = f"{args.episode_limit}_steps"
        suffix_total = f"{suffix_episode_limit}_{args.sb_code}_{args.standard}_{args.standard_mode}_{args.pv_ctrl}_{args.storage_p_ctrl}"
        self.extended_path = suffix_total

        # Generate config file path
        config_filename = (
            args.cfg_user_path.split(".")[0] + f"_{suffix_episode_limit}.yaml"
        )
        config_output_folder_name = os.path.join(config_folder_path, config_filename)

        if args.mpv_flag:
            additional1 = (
                f"_mpv_{args.mpv_flag}_con_{int(args.mpv_concentration_rate_percent)}"
            )
            self.extended_path = self.extended_path + additional1  # + additional2
            output_extended_path = os.path.join(output_dataset_path, self.extended_path)
        else:
            output_extended_path = os.path.join(output_dataset_path, self.extended_path)
        # Store updated paths in main configuration
        main_cfg["cfg"]["config_output_folder_name"] = config_output_folder_name
        main_cfg["cfg_settings"]["output_data_path"] = output_extended_path
        return main_cfg

    def update_arguments_from_master_script(self, main_cfg, args):
        """Update the main configuration with arguments from the main script.
        Args:
            main_cfg (dict): The main configuration dictionary.
            args (dict): Dictionary containing the keyword arguments.
        """
        main_cfg["rawdata"].update(
            {
                "benchmark": args.benchmark,
                "sb_code": args.sb_code,
                "scenario": args.scenario,
                "scaling": {
                    "pv": args.scaling_pv,
                    "load": args.scaling_load,
                    "storage": args.scaling_storage,
                },
                "soc_initial": args.soc_initial,
            }
        )
        main_cfg["sim"].update(
            {
                "time_mode": args.time_mode,
                "episode_start_hour": args.episode_start_hour,
                "episode_start_day": args.episode_start_day,
                "episode_start_min_interval": args.episode_start_min_interval,
                "episode_limit": args.episode_limit,
                "max_iterations": args.max_iterations,
            }
        )
        # Update default and user plot configurations
        main_cfg["cfg_settings"].update(
            {
                "run_main_base_path": args.base_path,
                "helper_path": os.path.join(args.base_path, "helper"),
                "cfg_user_plot_path": args.cfg_user_plot_path,
                "yaml_path": args.plattform,
                "output_data_path": os.path.join(args.output_path, self.extended_path),
                "cfg_default_plot_path": args.cfg_default_plot_path,
                "sequence_id_path": args.sequence_id_path,
            }
        )
        main_cfg["mcs_settings"].update(
            {
                "seed_value": args.seed_value,
                "flag_monte_carlo": args.flag_monte_carlo,
                "num_monte_carlo_runs": args.num_monte_carlo_runs,
            }
        )
        main_cfg["mpv_settings"].update(
            {
                "mpv_flag": args.mpv_flag,
                "mpv_benchmark": args.mpv_benchmark,
                "mpv_scaling": args.mpv_scaling,
                "mpv_concentration_rate_percent": args.mpv_concentration_rate_percent,
                "mpv_inverter_apparent_power_watt": args.mpv_inverter_apparent_power_watt,
                "mpv_solar_cell_capacity_watt": args.mpv_solar_cell_capacity_watt,
            }
        )
        # Print configuration status
        print(
            f"-------------------Configuration Settings-------------------\n"
            f"(i)   cfg_default settings from: \n{args.cfg_default_path} loaded...\n"
            f"(ii)  cfg_user settings from:    \n{args.cfg_user_path} loaded ...\n"
            f"(iii) merged (i) & (ii) in main_cfg:\n{main_cfg['cfg']['config_output_folder_name']}\n"
            f"(iv)  Results can be found in your local temp folder:\n"
            f"{main_cfg['cfg_settings']['output_data_path']}\n"
        )

    def set_control_modes(self, main_cfg, args):
        """Set valid control modes based on the main script arguments.
        Args:
            main_cfg (dict): The main configuration dictionary.
            args (dict): Dictionary containing the keyword arguments.
        """
        # Set valid control modes for timeseries, storage and pv control
        self.valid_modes = {
            "timeseries_ctrl": ["manual", "control_module", "test_mode"],
            "storage_p_ctrl": [
                "datasource",
                "rbc_pvbes_decentralized_sc_ctrl",
                "rbc_pvbes_distributed_sc_ctrl",  # rbc_solar_battery_ctrl
                "rbc_pvbes_distributed_sc_dnc_ctrl",
                "rbc_bes_dnc_ctrl",
            ],
            "storage_q_ctrl": [
                "datasource",
                "voltage_reactive_power_ctrl",
                "constant_power_factor_active_power_ctrl",
            ],
            "pv_ctrl": [
                "datasource",
                "voltage_reactive_power_ctrl",
                "power_factor_active_power_ctrl",
                "constant_power_factor_active_power_ctrl",
            ],
        }
        # Add argument: Timeseries, Storage and PV Control Mode
        self._set_control_mode(main_cfg, args.timeseries_ctrl, "timeseries_ctrl")
        self._set_control_mode(main_cfg, args.pv_ctrl, "pv_ctrl")
        self._set_control_mode(main_cfg, args.storage_p_ctrl, "storage_p_ctrl")
        self._set_control_mode(main_cfg, args.storage_q_ctrl, "storage_q_ctrl")
        # Print Args settings
        # ---------------------------------------------------------------------
        print("---------------------Arguments Settings---------------------")
        print(f"{'Setting':<35}{'Value'}")
        print(f"{'-'*60}")
        print(f"{'scaling_pv':<35}{args.scaling_pv}")
        print(f"{'scaling_load':<35}{args.scaling_load}")
        print(f"{'scaling_storage':<35}{args.scaling_storage}")
        print(f"{'timeseries_ctrl':<35}{args.timeseries_ctrl}")
        print(f"{'pv_ctrl':<35}{args.pv_ctrl}")
        print(f"{'storage_p_ctrl':<35}{args.storage_p_ctrl}")
        print(f"{'storage_q_ctrl':<35}{args.storage_q_ctrl}")
        # ---------------------------------------------------------------------

    def save_updated_configuration(self, main_cfg):
        """Create the Result folder and write the actual configuration to disk.
        Args:
            main_cfg (dict): The main configuration dictionary.
        """
        new_folder_name = main_cfg["cfg_settings"]["output_data_path"]
        os.makedirs(new_folder_name, exist_ok=True)
        with open(
            main_cfg["cfg"]["config_output_folder_name"], "w", encoding="utf-8"
        ) as filepath:
            yaml.dump(main_cfg, filepath)

    def load_inverter_control_settings(self, main_cfg, args):
        """Load inverter control settings based on the selected standard.
        Args:
            main_cfg (dict): Dictionary containing the main configuration settings.
            args (argparse.Namespace): Object containing the entered standards and modes.
        """
        # Load the parameters into the instance
        self.regulation_standard = args.standard.lower()
        self.standard_mode = args.standard_mode.lower()
        # Validate standard and standard_mode
        if self.regulation_standard not in main_cfg["regulation_standard_modes"]:
            raise ValueError(
                "Invalid standard. Choose 'vde' ('VDE-AR-N 4105')"
                " or 'ieee' ('IEEE Std 1547-2018')."
            )
        if self.standard_mode not in main_cfg["standard_modes"]:
            raise ValueError(
                f"Invalid standard mode. Choose one of the available modes"
                f"in {args.cfg_user_path}."
            )
        # Load the inverter control parameters
        self.inverter_ctrl_params = (
            main_cfg  # [self.regulation_standard] # [standard_mode]
        )
        # Print inverter control settings
        # ---------------------------------------------------------------------
        print(f"------------------Inverter Control Settings-----------------")
        print(f"{'standard:':<35}{self.regulation_standard.upper()}")
        print(f"{'standard_mode:':<35}{self.standard_mode.upper()}")

    def _merge_dicts(self, cfg_defaults, cfg_user):
        """Merge/update the user's configuration with default values where necessary.
        Merges/Updates the user's configuration with default values where necessary.
        Args:
            cfg_defaults (dict): Dictionary to iterate through.
            cfg_user (dict): Dictionary to update with missing keys.
        Notes:
           If a key in the default configuration is missing from the user's
           configuration, the default value is added, and a message is printed.
           If the value for a key in the default configuration is a dictionary,
           the function is called recursively to merge the sub-dictionaries,
           so missing keys are automatically added.
        """
        for key, default_value in cfg_defaults.items():
            if key not in cfg_user:
                cfg_user[key] = default_value
                print(f"default values are used: {key}:{default_value}")
            elif isinstance(default_value, dict):
                self._merge_dicts(default_value, cfg_user[key])

    def _set_control_mode(self, config, arg_ctrl_name, str_ctrl_name):
        """Set the control mode for a given argument.
        Args:
            config (dict): The configuration dictionary to update.
            arg_ctrl_name (str or None): The control mode specified as a command-line argument.
            str_ctrl_name (str): The name of the control mode to update in `config`.
            valid_modes (list): The list of valid control modes for the argument.
        Background:
            If the user has specified a value for timeseries control and it is
            included in the valid values, set timeseries_ctrl to that value,
            otherwise set it to the default value in main_cfg.
        Returns:
            None
        """
        if (
            arg_ctrl_name is not None
            and arg_ctrl_name in self.valid_modes[str_ctrl_name]
        ):
            ctrl_mode = arg_ctrl_name
        else:
            ctrl_mode = config["control_modes"][str_ctrl_name]
            print(
                f"----------Set control mode for a given argument----------\n"
                f"Control arguments ('{arg_ctrl_name}') from the main script"
                f"are not valid! Using default ctrl_mode: '{ctrl_mode}'"
            )
        # Set the value for timeseries control in main_cfg to ctrl_mode
        config["control_modes"][str_ctrl_name] = ctrl_mode

    def _check_config_settings(self):
        """Check the configuration dictionary `self.main_cfg` for errors and
        return a dictionary containing any errors found. If there are no errors,
        an empty dictionary is returned.
        Returns:
            dict: Dictionary containing any errors found in the configuration.
            If no errors are found, an empty dictionary is returned.
        Raises:
            AssertionError: If `time_mode` is not 'manual' or 'random'.
        """
        errors = {}
        assert self.main_cfg["sim"]["time_mode"] in [
            "selected",
            "random",
        ], '"time_mode" has to be in ["manual", "random"]!'
        if self.main_cfg["sim"]["episode_start_hour"] not in range(0, 25):
            errors["start_hour_error"] = "Please enter the start_hour between 0-24.!"
        if self.main_cfg["sim"]["episode_start_day"] not in range(0, 354):
            errors["start_day_error"] = "Please enter the start day between 0 and 354.!"
        if self.main_cfg["sim"]["episode_start_min_interval"] not in range(0, 4):
            errors["start_interval_error"] = (
                "Specify the starting interval between 0 and 3.!"
            )
        if self.main_cfg["sim"]["episode_limit"] not in range(0, 9600):
            errors["episode_limit_error"] = (
                "Specify the max. number of time steps in an episode.!"
            )
        if self.main_cfg["sim"]["max_iterations"] not in range(0, 96 * 30 * 12):
            errors["max_iterations_error"] = "Specify the max. number of iterations.!"
        if errors:
            for key, value in errors.items():
                print(f"key: {key}, value: {value}")
        return errors

    def _load_powergrid_network(self):
        """Load the power grid based on the selected benchmark configuration.
        Returns:
            A PandapowerNet object representing the loaded power grid.
        Raises:
            TypeError: If an invalid benchmark configuration is provided.
        Returns:
            Return a RootNet object based on the loaded power grid.
        """
        if self.benchmark == "simbench":
            # Load a simbench grid from a predefined config file.
            root_net = simbench.get_simbench_net(self.sb_code)
            # Save the loaded network to a file if specified in the config.
            if self.args["cfg_settings"]["write_to_rawdata"] is True:
                # absolute path
                root_net_path = os.path.join(self.input_data_path, "powergrid_net.p")
                pp.to_pickle(root_net, root_net_path)
        elif self.benchmark == "ToyGrid":
            root_net = toy_grid()
        elif self.benchmark == "customised":
            # absolute customised path
            root_net_path = os.path.join(self.input_data_path + "powergrid_net.p")
            root_net = pp.from_pickle(root_net_path)
        else:
            # Raise an error if an invalid benchmark configuration is provided.
            raise TypeError(
                f'Oops! That was no valid configuration file for benchmark: "{self.benchmark}".'
            )
        # Create and return a RootNet object based on the loaded power grid.
        return self._create_rootnet(root_net)

    def select_data_with_sampling_interval(
        data_path, header_rows, data_interval, desired_interval, scaling_factor
    ):
        """Selects data points with a desired sampling interval from a CSV file.
        Args:
            data_path (str): Path to the CSV file containing the data.
            header_rows (int): Number of header rows to skip.
            data_interval (int): Time interval between original data points in minutes.
            desired_interval (int): Desired sampling interval in minutes.
            scaling_factor (float): Scaling factor to apply to the selected data.
        Returns:
            pandas.DataFrame: Selected data points with the desired sampling
                              interval and scaled by the factor.
        """
        # Read data from CSV file
        time_series_data = pd.read_csv(data_path, index_col=None)
        time_series_data.index = pd.to_datetime(time_series_data.iloc[:, 0])
        time_series_data.index.name = "time"
        # Select data points with the desired sampling interval
        start_index = header_rows
        step_size = desired_interval // data_interval
        selected_data = (
            time_series_data.iloc[start_index::step_size, 1:] * scaling_factor
        )
        return selected_data

    def _load_mpv_data(self):
        """Load MPV data from a specified source
        ('mpvbench', 'simbench', or 'user-defined').
        This method applies adjustments and scales the data based on
        maximum solar power capacity(self.mpv_solar_cell_capacity_mw).
        Args:
            mpv_data (DataFrame): Data from 'simbench' source.
            self.mpv_solar_cell_capacity_mw (float): Maximum solar power capacity.
        Returns:
            mpv_data DataFrame: Modified MPV data.
        """
        # Code for loading from mpvbench
        if self.mpv_benchmark == "mpvbench":
            mpv_file_path = os.path.join(self.input_data_path, "mpvsgen\p_mw.xlsx")
            mpv_data = pd.read_excel(mpv_file_path, index_col=None)
            # drop first column that have been used as an index
            mpv_data.drop(mpv_data.columns[0], axis=1, inplace=True)
            # add uncertainty with unit truncated gaussian (only positive accepted)
            mpv_data = self.add_noise_to_data(
                data=mpv_data, add_noise=self.add_noise, std_dev_factor=10.0
            )
            print(mpv_data.max())
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            for column in mpv_data.columns:
                plt.plot(mpv_data[column], label=column)
        # Code for loading from simbench
        elif self.mpv_benchmark == "simbench":
            mpv_data = self.base_profiles[("sgen", "p_mw")] * self.mpv_scaling
            solar_cell_efficiency_rate = 0.95
            adjusted_solar_capacity_mw = (
                self.mpv_solar_cell_capacity_mw * solar_cell_efficiency_rate
            )
            calc_mpv_scaling = (mpv_data.max()) / (adjusted_solar_capacity_mw)
            # scale columns( Solar profiles) to max solar irradiation (self.mpv_solar_cell_capacity_mw capacity in MW).
            for column in mpv_data.columns:
                mpv_data[column] = mpv_data[column] / calc_mpv_scaling[column]
            # add uncertainty with unit truncated gaussian (only positive accepted)
            mpv_data = self.add_noise_to_data(
                data=mpv_data, add_noise=self.add_noise, std_dev_factor=10.0
            )
            # Calculate the maximum of each MPV data point and their mean
            max_mpv_data = mpv_data.max()
            max_avg_mpv_data = max_mpv_data.mean()
            # Calculate the mean and average of the mean of the MPV data
            avg_mpv_data = mpv_data.mean()
            avg_avg_mpv_data = avg_mpv_data.mean()
            # Number of timestamps in the MPV data
            num_timestamps = len(mpv_data)
            # Create an array filled with the average of the average MPV data
            avg_mpv_data_timestamps = np.full(num_timestamps, avg_avg_mpv_data)
            # Calculate the standard deviation of the MPV data
            std_mpv_data = mpv_data.std()
            avg_std_mpv_data = std_mpv_data.mean()
            avg_std_mpv_data_timestamps = np.full(num_timestamps, avg_std_mpv_data)
            # Display calculated statistical values
            print("---------------------- Load MPV Data -----------------------")
            print(f"{'max_avg_mpv_data:':<35}{max_avg_mpv_data:>2.6f} MW")
            print(f"{'Index':<4}{'Max Value':>15}{'Avg Value':>15}{'Std Value':>15}")
            print("-" * 60)
            # Display values for each timestamp
            for index in avg_mpv_data.index:
                max_val = max_mpv_data[index]
                avg_val = avg_mpv_data[index]
                std_val = std_mpv_data[index]
                print(f"{index:<4}{max_val:>15.6f}{avg_val:>15.6f}{std_val:>15.6f}")
            print("-" * 60)
            # Plotting the MPV data for visualization
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            for column in mpv_data.columns:
                plt.plot(mpv_data[column], label=column)
            # Add the line for the average MPV data
            plt.plot(
                avg_mpv_data_timestamps, label="avg mpv data", color="red", linewidth=2
            )
            # Plotting the confidence interval around the average MPV data
            upper_confidence = (
                avg_mpv_data_timestamps + avg_std_mpv_data_timestamps * 0.01
            )
            lower_confidence = (
                avg_mpv_data_timestamps - avg_std_mpv_data_timestamps * 0.01
            )
            plt.fill_between(
                range(num_timestamps),
                lower_confidence,
                upper_confidence,
                color="red",
                alpha=0.1,
            )
            # Adjustments for improving the plot presentation
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("MPV Value p_mw")
            plt.title("MPV Data Analysis")
            plt.show()
        else:
            raise TypeError(
                f"Oops! That was no valid configuration file for mpv_benchmark:"
                f'"{self.mpv_benchmark}".'
            )
        return mpv_data

    def _load_pv_data(self):
        """Default sampling frequency for the PV data is set at 15 minutes,
        with the flexibility to vary based on the raw sensor data."""
        if self.benchmark == "simbench":
            pv_data = self.base_profiles[("sgen", "p_mw")] * self.scaling_pv
            # add uncertainty with unit truncated gaussian (only positive accepted)
            pv_data = self.add_noise_to_data(
                data=pv_data, add_noise=self.add_noise, std_dev_factor=10.0
            )
            if self.args["cfg_settings"]["write_to_rawdata"] is True:
                pv_data.index = pd.to_datetime(
                    self.base_powergrid.profiles["renewables"]["time"]
                )
                output_path = os.path.join(
                    self.output_data_path,
                    f"pv_active_{self.benchmark}_{self.sb_code}_data.csv",
                )
                pv_data.to_csv(output_path)
        elif self.benchmark == "customised":
            pv_data_path = os.path.join(self.input_data_path, "pv_active.csv")
            pv_data = self.select_data_with_sampling_interval(
                data_path=pv_data_path,
                header_rows=4,
                data_interval=3,
                desired_interval=15,
                scaling_factor=self.scaling_pv,
            )
        else:
            raise TypeError(
                f'Oops! That was no valid configuration file for benchmark: "{self.benchmark}".'
            )
        return pv_data

    def _load_loads_active_power_data(self):
        """Default sampling frequency for the active load profile is 15 minutes,
        with the flexibility to vary based on the raw sensor data."""
        if self.benchmark == "simbench":
            load_active_power_data = (
                self.base_profiles[("load", "p_mw")] * self.scaling_load
            )
            # add uncertainty with unit truncated gaussian (only positive accepted)
            load_active_power_data = self.add_noise_to_data(
                data=load_active_power_data,
                add_noise=self.add_noise,
                std_dev_factor=100.0,
            )
            if self.args["cfg_settings"]["write_to_rawdata"] is True:
                load_active_power_data.index = pd.to_datetime(
                    self.base_powergrid.profiles["renewables"]["time"]
                )
                output_path = os.path.join(
                    self.output_data_path,
                    f"load_active_{self.benchmark}_{self.sb_code}_data.csv",
                )
                load_active_power_data.to_csv(output_path)
        elif self.benchmark == "customised":
            load_active_power_data_path = os.path.join(
                self.input_data_path, "load_active.csv"
            )
            load_active_power_data = self.select_data_with_sampling_interval(
                data_path=load_active_power_data_path,
                header_rows=4,
                data_interval=3,
                desired_interval=15,
                scaling_factor=self.scaling_load,
            )
        else:
            raise TypeError(
                f'Oops! Invalid configuration file for benchmark: "{self.benchmark}".'
            )
        return load_active_power_data

    def _load_loads_reactive_power_data(self):
        """Default sampling frequency for the reactive load profile is 15 minutes,
        with the flexibility to vary based on the raw sensor data."""
        if self.benchmark == "simbench":
            load_reactive_power_data = (
                self.base_profiles[("load", "q_mvar")] * self.scaling_load
            )
            # add uncertainty with unit truncated gaussian (only positive accepted)
            load_reactive_power_data = self.add_noise_to_data(
                data=load_reactive_power_data,
                add_noise=self.add_noise,
                std_dev_factor=100.0,
            )
            if self.args["cfg_settings"]["write_to_rawdata"] is True:
                load_reactive_power_data.index = pd.to_datetime(
                    self.base_powergrid.profiles["renewables"]["time"]
                )
                output_path = os.path.join(
                    self.output_data_path,
                    f"load_reactive_{self.benchmark}_{self.sb_code}_data.csv",
                )
                load_reactive_power_data.to_csv(output_path)
        elif self.benchmark == "customised":
            load_reactive_power_data_path = os.path.join(
                self.input_data_path, "load_reactive.csv"
            )
            load_reactive_power_data = self.select_data_with_sampling_interval(
                data_path=load_reactive_power_data_path,
                header_rows=4,
                data_interval=3,
                desired_interval=15,
                scaling_factor=self.scaling_load,
            )
        return load_reactive_power_data

    def _load_time_data(self):
        """Loads time data for the power grid simulation.
        If the benchmark is "simbench", the function retrieves time data from
        the loaded simbench power grid. The time data is then processed and
        formatted into a Pandas DataFrame with a minutely frequency.
        If the "write_to_rawdata" flag is set to True, the time data is saved
        as a CSV file in the input data directory.
        If the benchmark is "customised", the function does nothing and
        returns None.
        Raises:
            TypeError: If the benchmark is not "simbench" or "customised".
        Returns:
            A Pandas DataFrame containing the time data for the time series (ts)
            power grid simulation.
        """
        if self.benchmark == "simbench":
            time_data = self.base_powergrid.profiles["renewables"]["time"]
            start_string = time_data.values[0]
            end_string = time_data.values[-1]
            format_string = "%d.%m.%Y %H:%M"
            start_date = datetime.datetime.strptime(start_string, format_string)
            end_date = datetime.datetime.strptime(end_string, format_string)
            date_range = pd.date_range(
                start_date, end_date, freq=str(self.resolution) + "min"
            )  # minutely frequency
            time_data = pd.DataFrame({"timestamps": range(0, len(time_data))})
            # load time_data
            time_data = time_data.set_index(date_range)
            time_data.index.name = "time"
            if self.args["cfg_settings"]["write_to_rawdata"] is True:
                time_data.index = pd.to_datetime(
                    self.base_powergrid.profiles["renewables"]["time"]
                )
                output_path = os.path.join(
                    self.input_data_path,
                    f"time_data_{self.benchmark}_{self.sb_code}_data.csv",
                )
                time_data.to_csv(output_path)
        elif self.benchmark == "customised":
            pass
        else:
            raise TypeError(
                f'Oops! That was no valid configuration file for benchmark: "{self.benchmark}".'
            )
        return time_data

    def _load_price_data(self):
        """Default sampling frequency for the electricity market price is 15 minutes,
        with the flexibility to vary based on the raw sensor data."""
        # load_price_data = 0
        raise NotImplementedError("This method has not been implemented yet.")

    def _load_storage_data(self):
        """The sensor frequency is set to 15 minutes as default, with the
        flexibility to vary based on the raw sensor data.
        Notes:
        Since the storage power values are specified in the consumer system,
        positive power values model charging and
        negative power values model discharging.
        """
        if self.benchmark == "simbench":
            storage_active_power_data = self.base_profiles[("storage", "p_mw")]
            storage_reactive_power_data = (
                copy.deepcopy(self.base_profiles[("storage", "p_mw")]) * 0
            )
            if self.args["cfg_settings"]["write_to_rawdata"] is True:
                storage_active_power_data.index = pd.to_datetime(
                    self.base_powergrid.profiles["renewables"]["time"]
                )
                output_active_path = os.path.join(
                    self.output_data_path,
                    f"storage_active_{self.benchmark}_{self.sb_code}_data.csv",
                )
                storage_active_power_data.to_csv(output_active_path)
                storage_reactive_power_data.index = pd.to_datetime(
                    self.base_powergrid.profiles["renewables"]["time"]
                )
                output_reactive_path = os.path.join(
                    self.input_data_path,
                    f"storage_reactive_{self.benchmark}_{self.sb_code}_data.csv",
                )
                storage_reactive_power_data.to_csv(output_reactive_path)
        elif self.benchmark == "customised":
            storage_path = os.path.join(self.input_data_path, "load_storage.csv")
            storage = pd.read_csv(storage_path, index_col=None)
            storage.index = pd.to_datetime(storage.iloc[:, 0])
            storage.index.name = "time"
        else:
            raise TypeError(
                f'Oops! That was no valid configuration file for benchmark: "{self.benchmark}".'
            )
        return storage_active_power_data, storage_reactive_power_data

    def _load_and_apply_storage_constraints(self):
        """Assign actions to the simulation timestep based on the scenario.
        The scenario impacts time series raw data as well as the amount of
        elements in a distribution grid, specifically related to storage usage.
        """
        # Set network elements to load and sgen
        self.net_elements = ["load", "sgen"]
        if self.base_powergrid.storage.empty:
            self.is_storage_scenario = False
        if self.is_storage_scenario:
            # Load charging/discharging data for storage
            self.storage_active_power, self.storage_reactive_power = (
                self._load_storage_data()
            )
            # Set network elements to load, sgen, and storage
            self.net_elements = ["load", "sgen", "storage"]
            self.base_powergrid.storage["discharge_power_loss"] = 0.0

    def add_noise_to_data(self, data, add_noise, std_dev_factor):
        """Add Gaussian noise to the non-zero elements of the data.
        The standard deviation of the noise is determined by the standard
        deviation of the data divided by std_dev_factor.
        Parameters:
            data (pandas.DataFrame): Original data to which noise is added.
            add_noise (bool):       A flag to determine whether to add noise or not.
            std_dev_factor (float): A factor to scale the standard deviation of the data.
                                    The noise added is data.std(axis=0) / std_dev_factor.
        Returns:
            data (pandas.DataFrame):
                The data with noise added to non-zero elements.
                If add_noise is False, the original data is returned unchanged.
        """
        if add_noise:
            data_std = data.values.std(axis=0) / std_dev_factor
            noise = data_std * np.abs(np.random.randn(*data.shape))
            non_zero_indices = np.where(data != 0)
            data.values[non_zero_indices] += noise[non_zero_indices]
        return data

    def get_zone_data(self):
        """Calculate branch paths and zones for a radiation network using the
        Dijkstra algorithm.
        Args:
            **net** (pandapowerNet) -
                Variable that contains a pandapower network that represents the
                power grid on which the branch paths and zones are to be calculated.
        Returns:
            **zone_data** (dict) -
                Dictionary containing the calculated zones and their corresponding nodes.
                The keys are the names of the zones and the values are lists of
                buses that belong to each zone. The zones include the main zone
                and any additional zones that are created by the function.
        """
        number_of_transformers = len(self.base_powergrid.trafo)
        if number_of_transformers > 1:
            raise NotImplementedError(
                "Function not available for nets with multiple transformers. "
                "Please use nets with only one transformer."
            )
        source_node = self.base_powergrid.trafo.lv_bus.values[0]
        graph = create_nxgraph(
            self.base_powergrid, respect_switches=True, nogobuses=None, notravbuses=None
        )
        # predefined vars
        # branch_paths = []
        bus_distance = pd.Series(
            nx.single_source_dijkstra_path_length(graph, source_node, weight="weight")
        )
        zone_data = {}
        iteration_count = 0
        # while len(list(graph.nodes)) > 3:
        while not (bus_distance.values == 0).all():
            target_node = bus_distance[bus_distance == bus_distance.max()].index[0]
            # Get all simple paths between start and end point
            paths = list(nx.all_simple_paths(graph, source_node, target_node))
            # branch_paths.append(paths[0])
            zone = {f"zone{iteration_count+1}": paths[0][1:]}
            zone_data.update(zone)
            graph.remove_nodes_from(paths[0][1:])
            selected_rows = self.base_powergrid.bus.index.isin(
                zone_data[f"zone{iteration_count+1}"]
            )
            self.base_powergrid.bus.loc[selected_rows, "zone"] = (
                f"zone{iteration_count+1}"
            )
            bus_distance = pd.Series(
                nx.single_source_dijkstra_path_length(
                    graph, source_node, weight="weight"
                )
            )
            iteration_count += 1
        zone_data["main"] = list(graph.nodes)
        self.base_powergrid.bus.loc[
            self.base_powergrid.bus.index.isin(zone_data["main"]), "zone"
        ] = "main"
        pp.add_zones_to_elements(self.base_powergrid, elements=self.net_elements)
        return zone_data

    def _get_penetration_rates(self):
        """Performs a power grid simulation analysis, calculates penetration rates,
        and saves the results to a text file."""
        # Initialize simulation parameters
        interval_per_hour = (
            60 // self.resolution
        )  # (each interval is a fixed length of time)
        start = (
            self.episode_start_min_interval
            + (self.episode_start_hour * interval_per_hour)
            + (self.episode_start_day * 24 * interval_per_hour)
        )
        # Determine the total number of intervals needed for the episode plus history
        num_intervals = self.episode_limit  # + self.history + 1 # (with a margin of 1)
        # Create a list of intervals that will be used for the simulation
        self.base_sim_steps = range(start, start + num_intervals)
        end_step, start_step = max(self.base_sim_steps), min(self.base_sim_steps)
        self.sim_steps = range(0, (end_step - start_step) + 1)

        # Slicing the data for the simulation interval
        part_pv_data = self.pv_data.iloc[start_step:end_step]
        part_load_active_data = self.load_active_power_data.iloc[start_step:end_step]
        part_load_reactive_data = self.load_reactive_power_data.iloc[
            start_step:end_step
        ]

        # Calculating sum data for the interval
        pv_sim_sum = part_pv_data.sum().sum()
        load_active_sim_sum = part_load_active_data.sum().sum()
        load_reactive_sim_sum = part_load_reactive_data.sum().sum()
        apparent_load_power_sim_sum = np.sqrt(
            load_active_sim_sum**2 + load_reactive_sim_sum**2
        )

        # Calculating penetration rates
        pv_penetration_rate = (
            pv_sim_sum / load_active_sim_sum if load_active_sim_sum != 0 else 0
        )
        pv_penetration_rate2 = (
            pv_sim_sum / apparent_load_power_sim_sum
            if apparent_load_power_sim_sum != 0
            else 0
        )

        mpv_sim_sum, mpv_penetration_rate, mpv_penetration_rate2 = 0, 0, 0
        if self.mpv_flag:
            # Load MPV data (Mini Photovoltaic energy generation)
            self.mpv_data = self._load_mpv_data()
            part_mpv_data = self.mpv_data.iloc[start_step:end_step]
            mpv_sim_sum = part_mpv_data.sum().sum()
            mpv_penetration_rate = (
                mpv_sim_sum / load_active_sim_sum if load_active_sim_sum != 0 else 0
            )
            mpv_penetration_rate2 = (
                mpv_sim_sum / apparent_load_power_sim_sum
                if apparent_load_power_sim_sum != 0
                else 0
            )

        # Printing the results
        print("-------------------  Penetration PV\MPV  -------------------")
        print(
            f"{'Total Apparent Power Load (S) in Simulation:':<60}{apparent_load_power_sim_sum:.2f} MVA"
        )
        print(
            f"{'Total Active Power Load (P) in Simulation:':<60}{load_active_sim_sum:.2f} MW"
        )
        print(
            f"{'Total Reactive Power Load (Q) in Simulation:':<60}{load_reactive_sim_sum:.2f} MVar"
        )
        print(
            f"{'Total Active Power Generation from PV (P) in Simulation:':<60}{pv_sim_sum:.2f} MW"
        )
        print(
            f"{'Total Active Power Generation from MPV (P) in Simulation:':<60}{mpv_sim_sum:.2f} MW"
        )
        print(f"{'PV Penetration Rate (PR):':<60}{pv_penetration_rate:.2f}")
        print(f"{'PV Penetration Rate (PR2):':<60}{pv_penetration_rate2:.2f}")
        print(f"{'MPV Penetration Rate (PR):':<60}{mpv_penetration_rate:.2f}")
        print(f"{'MPV Penetration Rate (PR2):':<60}{mpv_penetration_rate2:.2f}")
        # Storing the data in a dictionary

        # Storing the data in a dictionary
        self.sim_info_data = {
            "total_apparent_power_load_MVA": apparent_load_power_sim_sum,
            "total_active_power_load_MW": load_active_sim_sum,
            "total_reactive_power_load_MVar": load_reactive_sim_sum,
            "total_active_power_generation_pv_MW": pv_sim_sum,
            "total_active_power_generation_mpv_MW": mpv_sim_sum,
            "pv_penetration_rate": pv_penetration_rate,
            "pv_penetration_rate2": pv_penetration_rate2,
            "mpv_penetration_rate": mpv_penetration_rate,
            "mpv_penetration_rate2": mpv_penetration_rate2,
        }
        # Saving the data to a text file
        sim_info_file_name = "sim_info_data.txt"
        sim_info_path = os.path.join(self.output_data_path, sim_info_file_name)
        with open(sim_info_path, "w") as file:
            for key, value in self.sim_info_data.items():
                file.write(f"{key}: {value}\n")
        # Confirming the data is saved
        print(f"{'Penetration Info: successfully':<60}")
        print(f"{'saved in':<60}{sim_info_path}")

    def _valid_and_set_scenario_name(self):
        """Set the scenario based on simbench data and validate the scenario name.
        This digit is mapped to one of three possible scenarios:
        - 0: "today" (normal real today case)
        - 1: "near future" (normal DERs increase)
        - 2: "future" (large DERs increase, including warm pumps, electromobilities)
        - 3: "customised" (planned own scenario with Solar Mini Balcony Power Plant)
        https://www.vde.com/de/presse/pressemitteilungen/2023-01-11-mini-pv
        https://www.vde.com/resource/blob/2229846/acbd1078371f6a553a049a1d33b8612c/positionspapier-data.pdf
        The scenario is assigned to the `scenario` attribute and information
        is printed to the console.
        If `benchmark` is not "simbench", `scenario` is set to "customised" and
        a message prompts the user to define a custom scenario
        (not yet implemented).
        Returns attributes:
            self.scenario (str): The name of the scenario
                 "today", "near future", "future", or "customised".
        Raises:
            AssertionError: If sb_code doesn't match valid_sb_code.
        """
        # Generate valid simbench scenario codes, including "customised".
        self.valid_sb_code = [
            code.format(i) for code in self.valid_sb_base_codes for i in range(4)
        ] + ["customised"]
        # Break down sb_code and replace the second part with the first character of the scenario.
        parts = self.sb_code.split("--")
        num_scenario = self.scenario.split("-")[0]
        parts_one = parts[0]
        parts_two = num_scenario + parts[1][1:]
        self.sb_code = "--".join([parts_one, parts_two])
        # Check if the sb_code is valid.
        assert self.sb_code in self.valid_sb_code, f"Invalid sb_code: {self.sb_code}"
        # If benchmark is simbench, find the matching scenario and print out information.
        if self.benchmark == "simbench":
            for scenario in self.args["rawdata"]["valid_sb_scenario"]:
                if scenario == self.scenario:
                    print(
                        "----------------------- Console Output ---------------------"
                    )
                    print(f"{'Benchmark:':<35}{self.benchmark}")
                    print(f"{'Scenario:':<35}{self.scenario}")
                    print(f"{'Powergrid-Code:':<35}{self.sb_code}")
        else:
            # If benchmark is not simbench, set scenario to "customised" and print out information.
            self.scenario = "customised"
            print(
                f"----------------------- Console Output ---------------------\n"
                f"Scenario {self.benchmark}:  {self.scenario}\n"
                f"Powergrid-Code: Custom arbitrary must be defined by the user\n"
            )
        # Determine if the scenario involves storage elements.
        self.is_storage_scenario = self.scenario in self.valid_sb_scenario_storage
        print(f"{'Is Storage Scenario:':<35}{self.is_storage_scenario}")
        # Determine if the scenario involves mpv elements.
        print(f"{'Is MPV Scenario:':<35}{self.mpv_flag}")

    def _create_rootnet(self, root_net):
        """initilization of power grid set the pandapower net to use"""
        if root_net is None:
            raise TypeError(
                "Please provide a base powergrid configured as pandapower format."
            )
        return root_net

    # load
    def get_res_load_active(self):
        """Returns the (active power of the load) in the power grid.
        :return: NumPy array of active power values in (load.p_mw) [MW]."""
        load_active_power = self.net.res_load.p_mw.values
        return load_active_power

    def get_res_load_reactive(self):
        """Returns the (reactive power of the load) in the power grid.
        :return: NumPy array of reactive power values in (load.q_mvar) [MVar]."""
        load_reactive_power = self.net.res_load.q_mvar.values
        return load_reactive_power

    # sgen
    def get_sgen_active(self):
        """Returns the (active power of the static generator) in the power grid.
        :return: NumPy array of active power values in (sgen.p_mw) [MW]."""
        pv_active = self.net.sgen.p_mw.to_numpy(copy=True)
        return pv_active

    def get_sgen_reactive(self):
        """Returns the (reactive power demand of the sgen) in the power grid.
        :return: NumPy array of reactive power values in (sgen.q_mvar) [MVar]."""
        pv_reactive = self.net.sgen.q_mvar.to_numpy(copy=True)
        return pv_reactive

    # bus
    def get_res_bus_active(self):
        """Returns the (active power demand) at the buses in the power grid.
        :return: NumPy array of active power values in (res_bus.p_mw) [MW]."""
        res_bus_active = self.net.res_bus.p_mw.sort_index().to_numpy(copy=True)
        return res_bus_active

    def get_res_bus_reactive(self):
        """Returns the (resulting reactive power demand) at the buses in the power grid.
        :return: NumPy array of reactive power values in (res_bus.q_mvar) [Mvar]."""
        res_bus_reactive = self.net.res_bus.q_mvar.sort_index().to_numpy(copy=True)
        return res_bus_reactive

    def get_res_bus_voltage(self):
        """Returns the voltage magnitude at the buses in the power grid.
        :return: NumPy array of voltage magnitudes in (res_bus.vm_pu) [p.u]."""
        res_bus_voltage = self.net.res_bus.vm_pu.sort_index().to_numpy(copy=True)
        return res_bus_voltage

    def get_res_bus_voltage_angle(self):
        """Returns the (voltage angle at the buses) in the power grid.
        :return: NumPy array of voltage angles in (res_bus.va_degree) [degree]."""
        res_bus_voltage_angle = self.net.res_bus.va_degree.sort_index().to_numpy(
            copy=True
        )
        return res_bus_voltage_angle

    # line
    def get_res_line_loss(self):
        """Returns the (active power losses of the lines) in the power grid.
        :return: NumPy array of active power losses in (res_line.pl_mw) [MW]."""
        res_line_loss = self.net.res_line.pl_mw.sort_index().to_numpy(copy=True)
        return res_line_loss

    def get_res_line_loading(self):
        """Returns the (loading percentage of the lines) in the power grid.
        :return: NumPy array of line loading percentages in (res_line.loading_percent) [%].
        """
        res_line_loading = self.net.res_line.loading_percent.sort_index().to_numpy(
            copy=True
        )
        return res_line_loading

    # trafo
    def get_res_trafo_loading(self):
        """Returns the loading percentage of the transformers in the power grid.
        :return: NumPy array of transformer loading percentages in
        (res_trafo.loading_percent): trafo loading [%]."""
        res_trafo_loading = self.net.res_trafo.loading_percent.sort_index().to_numpy(
            copy=True
        )
        return res_trafo_loading

    def get_res_trafo_active(self):
        """Returns the active power flow at the high voltage bus of the
        transformers in the power grid.
        :return: NumPy array of (active power flow at the high voltage
                                 transformer bus) in (res_trafo.p_hv_mw) [MW]."""
        res_trafo_active = self.net.res_trafo.p_hv_mw.sort_index().to_numpy(copy=True)
        return res_trafo_active

    # ext_grid
    def get_res_ext_grid_active(self):
        """Returns the active power supply at the external grid in the power grid.
        :return: NumPy array of (active power supply at the external grid)
        in (res_ext_grid.p_mw) [MW]."""
        res_ext_grid_active = self.net.res_ext_grid.p_mw.sort_index().to_numpy(
            copy=True
        )
        return res_ext_grid_active

    def get_res_ext_grid_reactive(self):
        """Returns the reactive power supply at the external grid in the power grid.
        :return: NumPy array of (reactive power supply at the external grid)
                 in (res_ext_grid.q_mvar) [MVar]."""
        res_ext_grid_reactive = self.net.res_ext_grid.q_mvar.sort_index().to_numpy(
            copy=True
        )
        return res_ext_grid_reactive

    # storage
    def get_storage_active(self):
        """Returns the (momentary real power of the storage) in the power grid.
        Positive values indicate charging and negative values indicate discharging.
        :return: NumPy array of momentary real power of the energy storage in
        (storage.p_mw) [MW].
        """
        storage_active = self.net.storage.p_mw.sort_index().to_numpy(copy=True)
        return storage_active

    def get_storage_soc_percent(self):
        """Returns the state of charge of the energy storage in the power grid.
        :return: NumPy array of (state of charge of the energy storage)
                in (storage.soc_percent) [%]. [0% =< soc_percent =< 100%]"""
        storage_soc_percent = self.net.storage.soc_percent.sort_index().to_numpy(
            copy=True
        )
        return storage_soc_percent

    def get_generate_random_numbers(self, size):
        """Generate an array of random numbers.
        Parameters: size (int or tuple): Shape of the returned array.
        Returns: numpy.ndarray: Array of random numbers of given 'size'."""
        return np.random.random(size)

    def print_logo_iai_kit(self):
        """Prints the IAI KIT logo to the console."""
        logo = r"""
        **********************************************
        *    _  __ ___  _____   ___     _     ___    *
        *   | |/ /|_ _||_   _| |_ _|   / \   |_ _|   *
        *   | ' /  | |   | |    | |   / _ \   | |    *
        *   | . \  | |   | |    | |  / ___ \  | |    *
        *   |_|\_\|___|  |_|   |___|/_/   \_\|___|   *
        **********************************************
        *                    PIDE                    *
        **********************************************
        *          Grid Control Simulation           *
        **********************************************
        """
        print(logo)
