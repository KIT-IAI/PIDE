"""
helper_powergrid_advanced_plot.py
---------------------------------
This module provides advanced plotting and configuration management functionalities for power grid
visualization using pandapower and SimBench. It includes tools for loading grid configurations,
applying advanced plotting techniques, generating plots for various scenarios for documentation.
"""

# Importing necessary libraries
import os
import sys
import numpy as np
import pandas as pd
import simbench as sb
import pandapower as pp
import pandapower.plotting as plot

import matplotlib.pyplot as plt
import random
import yaml
import logging

logger = logging.getLogger(__name__)


class PlotConfigManager:
    def __init__(self, base_path, yaml_path):
        self.base_path = base_path
        self.yaml_path = yaml_path
        self.parent_path = os.path.dirname(base_path)
        # Set the configuration file names and paths
        self.cfg_default_path = os.path.join(
            self.parent_path,
            "yaml",
            self.yaml_path,
            "default_config_grid_manage_der.yaml",
        )
        self.cfg_user_path = os.path.join(
            self.parent_path, "yaml", self.yaml_path, "user_config_grid_manage_der.yaml"
        )
        # Load default and user configurations
        self.cfg_default = self.load_configuration_file(self.cfg_default_path)
        self.cfg_user = self.load_configuration_file(self.cfg_user_path)
        self.main_config = self.cfg_user
        # Set simbench episode settings
        self._set_simbench_episode_settings()
        # Define the path for the configuration file
        self.raw_cfg_user_plot_path = self.main_config["cfg_settings"][
            "cfg_user_plot_path"
        ]
        self.cfg_user_plot_path = os.path.join(self.parent_path, "yaml", self.yaml_path)
        self.cfg_plot_yaml_path = os.path.join(
            self.cfg_user_plot_path, self.raw_cfg_user_plot_path
        )
        self.plot_settings = self.load_plot_configuration(self.cfg_plot_yaml_path)

        # Access various plot plot_configuration parameters
        self.basic_plot_params = self.plot_settings[
            "basic_plot"
        ]  # 6 LV Grids for every scenario
        self.simple_plot_params = self.plot_settings[
            "simple_plot"
        ]  # 6 LV Grids for every scenario (load,sgen)
        self.advanced_plot_params = self.plot_settings[
            "advanced_plot"
        ]  # 6 LV Grids for every scenario (load,sgen,storage)
        self.multi_plot_params = self.plot_settings["multi_plot_settings"]

        # Sets the parameters for the Mini Photovoltaic (MPV) system
        self.mpv_concentration_rate_percent = self.plot_settings[
            "mpv_concentration_rate_percent"
        ]
        self.mpv_inverter_apparent_power_mva = self.convert_watt_to_megawatt(
            self.plot_settings["mpv_inverter_apparent_power_watt"]
        )
        self.mpv_solar_cell_capacity_mw = self.convert_watt_to_megawatt(
            self.plot_settings["mpv_solar_cell_capacity_watt"]
        )

    def load_configuration_file(self, cfg_filename):
        """
        Load a configuration file.
        Args:
            cfg_filename (str): Path to the configuration file.
        Returns:
            dict: Dictionary containing the configuration settings.
        """
        with open(cfg_filename, encoding="utf-8") as stream:
            try:
                return yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                logger.error(f"Error in configuration file: {exc}, {stream.name}")
                sys.exit()

    def load_plot_configuration(self, cfg_plot_path):
        # Load the configuration from the YAML file
        with open(cfg_plot_path, "r") as file:
            plot_settings = yaml.safe_load(file)
        # Update specific plot_configuration parameters
        plot_settings["multi_plot_settings"]["num_scenarios"] = len(
            self.valid_sb_scenario
        )
        plot_settings["multi_plot_settings"]["num_codes"] = len(
            self.valid_sb_base_codes
        )
        # Update paths in the plot_configuration
        plot_settings["path"]["base_path"] = self.base_path
        plot_settings["path"]["cfg_user_plot_path"] = cfg_plot_path
        self.plot_settings = plot_settings
        self.update_plot_configuration()
        return self.plot_settings

    def update_plot_configuration(self):
        self.plot_settings["path"]["save_output_plot_path"] = (
            self.plot_settings["path"]["parent_path"]
            + self.plot_settings["path"]["output_plot_path"]
        )
        self.plot_settings["path"]["save_output_latex_path"] = (
            self.plot_settings["path"]["parent_path"]
            + self.plot_settings["path"]["output_latex_path"]
        )
        # Write the updated plot_configuration back to the YAML file
        with open(self.cfg_plot_yaml_path, "w") as file:
            yaml.dump(self.plot_settings, file, default_flow_style=False)

    def _set_simbench_episode_settings(self):
        """Set the simbench mode, simulation code, and simulation parameters for
        an episode."""
        # Set the simbench mode and code for the simulation.
        self.benchmark = self.main_config["rawdata"]["benchmark"]
        self.sb_code = self.main_config["rawdata"]["sb_code"]
        self.scenario = self.main_config["rawdata"]["scenario"]
        # Set the sb_code and scenario for the simulation.
        self.valid_sb_code = self.main_config["rawdata"]["valid_sb_code"]
        self.valid_sb_base_codes = self.main_config["rawdata"]["valid_sb_base_codes"]
        self.valid_sb_scenario = self.main_config["rawdata"]["valid_sb_scenario"]
        self.valid_sb_scenario_storage = self.main_config["rawdata"][
            "valid_sb_scenario_storage"
        ]
        # Set scenario name
        self._valid_and_set_scenario_name()

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
            code.format(i) for code in self.valid_sb_base_codes for i in range(3)
        ] + ["customised"]
        # Break down sb_code and replace the second part with the first character of the scenario.
        parts = self.sb_code.split("--")
        self.sb_code = "--".join([parts[0], self.scenario[0] + parts[1][1:]])
        # Check if the sb_code is valid.
        assert self.sb_code in self.valid_sb_code, f"Invalid sb_code: {self.sb_code}"
        # If benchmark is simbench, find the matching scenario and print out information.
        if self.benchmark == "simbench":
            for scenario in self.valid_sb_scenario:
                if scenario == self.scenario:
                    print(
                        f"----------------------- Console Output ---------------------\n"
                    )
                    print(f"{'Benchmark:':<35}{self.benchmark}")
                    print(f"{'Scenario:':<35}{self.scenario}")
                    print(f"{'Powergrid-Code:':<35}{self.sb_code}")
        else:
            # If benchmark is not simbench, set scenario to "customised" and print out information.
            self.scenario = "customised"
            print(f"----------------------- Console Output ---------------------\n")
            print(f"{'Benchmark:':<35}{self.benchmark}")
            print(f"{'Scenario:':<35}{self.scenario}")
            print(f"Powergrid-Code: Custom arbitrary must be defined by the user\n")
        # Determine if the scenario involves storage elements.
        self.is_storage_scenario = self.scenario in self.valid_sb_scenario_storage
        print(f"{'Is Storage Scenario:':<35}{self.is_storage_scenario}")

    # 3. Hilfsfunktionen
    def merge(self, sb_code, sb_scenario):
        """Merges SimBench code and scenario."""
        parts = sb_code.split("--")
        scenario_number = sb_scenario.split("-")[0]
        new_code = "--".join([parts[0], scenario_number + parts[1][2:]])
        return new_code

    def get_simbench_code(self, code_index, scenario_index):
        """Retrieves selected SimBench code."""
        # Check if indices are valid
        if (code_index < 0 or code_index >= len(self.valid_sb_base_codes)) or (
            scenario_index < 0 or scenario_index >= len(self.valid_sb_scenario)
        ):
            raise ValueError("Invalid indices provided.")
        sb_code_template = self.valid_sb_base_codes[code_index]
        sb_scenario = self.valid_sb_scenario[scenario_index]
        sb_code_merged = self.merge(sb_code_template, sb_scenario)
        print(sb_code_merged)
        return sb_code_merged

    def get_collection_sizes(
        self,
        net,
        bus_size=1.0,
        ext_grid_size=1.0,
        trafo_size=1.0,
        load_size=1.0,
        sgen_size=1.0,
        mpv_sgen_size=1.0,
        storage_size=1.0,
        annotation_size=1.0,
        factor=10.0,
        distance_factor=200.0,
    ):
        """
        Calculates the size for most collection types according to the distance between min and max
        geocoord so that the collections fit the plot nicely
    
        .. note: This is implemented because if you would choose a fixed values (e.g. bus_size = 0.2),\
            the size could be to small for large networks and vice versa
        """
        mean_distance_between_buses = sum(
            (
                net["bus_geodata"].loc[:, ["x", "y"]].max()
                - net["bus_geodata"].loc[:, ["x", "y"]].min()
            ).dropna()
            / distance_factor
        )
        sizes = {
            "bus": bus_size * mean_distance_between_buses * factor,
            "annotation": annotation_size * mean_distance_between_buses * factor * 0.5,
            "ext_grid": ext_grid_size * mean_distance_between_buses * 1.5 * factor,
            "storage": storage_size * mean_distance_between_buses * 1 * factor,
            "load": load_size * mean_distance_between_buses * factor,
            "sgen": sgen_size * mean_distance_between_buses * factor,
            "mpv_sgen": mpv_sgen_size * mean_distance_between_buses * factor,
            "trafo": trafo_size * mean_distance_between_buses * factor,
        }
        print(sizes)
        return sizes

    @staticmethod
    def convert_watt_to_megawatt(watt_value):
        """Converts a value from watts to megawatts.
        Args: watt_value: The value in watts to be converted.
        Returns: The value in megawatts.
        """
        megawatt_value = watt_value / 1e6
        return megawatt_value

    # 4. Hauptfunktionen
    def load_simbench_powergrid(
        self, code_index, scenario_index, from_disk=False, rotate_plot=False
    ):
        """
        Creates or loads a SimBench network based on the provided code and scenario indices.
        Args:
        code_index (int): Index to select the SimBench code.
        scenario_index (int): Index to select the scenario.
        from_disk (bool): If True, load the network from an Excel file. Default is False.
        rotate_plot (bool): If True, swap x and y axis to rotate the plot. Default is False.
        Returns:
        net (pandapowerNet): The SimBench network after running the power flow.
        """
        if from_disk:
            net = pp.from_excel(os.path.join("data", "case_study_grid.xlsx"))
        else:
            simbench_code = self.get_simbench_code(code_index, scenario_index)
            print(simbench_code)
            net = sb.get_simbench_net(simbench_code)
            print(f"sb code: {simbench_code}")
            print(net)
            plot.create_generic_coordinates(net, respect_switches=False, overwrite=True)
            if rotate_plot:
                # swap x and y axis to rotate plot
                net.bus_geodata.columns = list(net.bus_geodata.columns)[::-1]
        return net, simbench_code

    # 5. Plotting-Funktionen
    def advanced_plot(
        self,
        net,
        respect_switches=False,
        line_width=1.0,
        bus_size=1.0,
        ext_grid_size=1.0,
        trafo_size=1.0,
        load_size=1.0,
        sgen_size=1.0,
        mpv_sgen_size=1.0,
        storage_size=1.0,
        annotation_size=1.0,
        factor=1.0,
        distance_factor=100.0,
        mpv_concentration_rate_percent=100.0,
        scale_size=True,
        plot_loads=False,
        plot_sgens=False,
        plot_mpv_sgens=False,
        plot_storages=False,
        plot_annotation=False,
        orientation_loads=1,
        orientation_sgens=(25 / 32),
        orientation_mpv_sgens=(10 / 16),
        orientation_storage=(5 / 4),
        bus_color="b",
        line_color="grey",
        trafo_color="k",
        ext_grid_color="k",
        sgen_color="k",
        storage_color="k",
        library="igraph",
        show_plot=True,
        ax=None,
    ):
        """               
        Plots a pandapower network as simple as possible. If no geodata is available, artificial
        geodata is generated. For advanced plotting see the tutorial
        INPUT:
            **net** - The pandapower format network.
        OPTIONAL:
            **line_width** (float, 1.0) - width of lines
            **bus_size** (float, 1.0) - Relative size of buses to plot.
                                        The value bus_size is multiplied with mean_distance_between_buses, which equals the
                                        distance between
                                        the max geoocord and the min divided by 200.
                                        mean_distance_between_buses = sum((net['bus_geodata'].max() - net['bus_geodata'].min()) / 200)
            **ext_grid_size** (float, 1.0) - Relative size of ext_grids to plot. See bus sizes for details.
                                                Note: ext_grids are plottet as rectangles
            **trafo_size** (float, 1.0) - Relative size of trafos to plot.
            **load_size** (float, 1.0) - Relative size of loads to plot.
            **sgen_size** (float, 1.0) - Relative size of sgens to plot.
            **scale_size** (bool, True) - Flag if bus_size, ext_grid_size, bus_size- and distance \
                                          will be scaled with respect to grid mean distances
            **plot_loads** (bool, False) - Flag to decide whether load symbols should be drawn.
            **plot_gens** (bool, False) - Flag to decide whether gen symbols should be drawn.
            **plot_sgens** (bool, False) - Flag to decide whether sgen symbols should be drawn.
            **plot_storages** (bool, False) - Flag to decide whether storage symbols should be drawn.
            **bus_color** (String, colors[0]) - Bus Color. Init as first value of color palette. Usually colors[0] = "b".
            **line_color** (String, 'grey') - Line Color. Init is grey
            **trafo_color** (String, 'k') - Trafo Color. Init is black
            **ext_grid_color** (String, 'y') - External Grid Color. Init is yellow
            **sgen_color** (String, 'k') - Sgen Color. Init is black
            **storage_color** (String, 'k') - Storage Color. Init is black
            **library** (String, "igraph") - library name to create generic coordinates (case of
                                                missing geodata). "igraph" to use igraph package or "networkx" to use networkx package.
            **show_plot** (bool, True) - Shows plot at the end of plotting
            **ax** (object, None) - matplotlib axis to plot to
        OUTPUT:
            **ax** - axes of figure
        """
        respect_switches = True
        library = "igraph"
        geodata_table = "bus_geodata"  # net["bus_geodata"]
        net[geodata_table].drop(net[geodata_table].index, inplace=True)
        net[geodata_table] = pd.DataFrame(columns=["x", "y"])
        # create geocoord if none are available
        if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
            logger.warning(
                "No or insufficient geodata available --> Creating artificial coordinates."
                + " This may take some time"
            )
            plot.generic_geodata.create_generic_coordinates(
                net, respect_switches=respect_switches, library=library, overwrite=True
            )
        if scale_size:
            # if scale_size -> calc size from distance between min and max geocoord
            sizes = self.get_collection_sizes(
                net,
                bus_size,
                ext_grid_size,
                trafo_size,
                load_size,
                sgen_size,
                mpv_sgen_size,
                storage_size,
                annotation_size,
                factor=factor,
                distance_factor=distance_factor,
            )
            bus_size = sizes["bus"]
            ext_grid_size = sizes["ext_grid"]
            trafo_size = sizes["trafo"]
            load_size = sizes["load"]
            sgen_size = sizes["sgen"]
            mpv_sgen_size = sizes["mpv_sgen"]
            storage_size = sizes["storage"]
            annotation_size = sizes["annotation"]

        # create bus collections to plot
        bc = plot.create_bus_collection(
            net, net.bus.index, size=bus_size, color=bus_color, zorder=10
        )
        # if bus geodata is available, but no line geodata
        use_bus_geodata = len(net.line_geodata) == 0
        in_service_lines = net.line[net.line.in_service].index
        nogolines = (
            set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)])
            if respect_switches
            else set()
        )
        plot_lines = in_service_lines.difference(nogolines)

        # create line collections
        lc = plot.create_line_collection(
            net,
            plot_lines,
            color=line_color,
            linewidths=line_width,
            use_bus_geodata=use_bus_geodata,
        )
        collections = [bc, lc]
        # create ext_grid collections
        # eg_buses_with_geo_coordinates = set(net.ext_grid.bus.values) & set(net.bus_geodata.index)
        if len(net.ext_grid) > 0:
            ext_gridc = plot.create_ext_grid_collection(
                net,
                size=ext_grid_size,
                orientation=0,
                ext_grids=net.ext_grid.index,
                patch_edgecolor=ext_grid_color,
                zorder=11,
            )
            collections.append(ext_gridc)

        # create trafo collection if trafo is available
        trafo_buses_with_geo_coordinates = [
            t
            for t, trafo in net.trafo.iterrows()
            if trafo.hv_bus in net.bus_geodata.index
            and trafo.lv_bus in net.bus_geodata.index
        ]
        if len(trafo_buses_with_geo_coordinates) > 0:
            tc = plot.create_trafo_collection(
                net,
                trafo_buses_with_geo_coordinates,
                color=trafo_color,
                size=trafo_size,
            )
            collections.append(tc)

        # create trafo3w collection if trafo3w is available
        trafo3w_buses_with_geo_coordinates = [
            t
            for t, trafo3w in net.trafo3w.iterrows()
            if trafo3w.hv_bus in net.bus_geodata.index
            and trafo3w.mv_bus in net.bus_geodata.index
            and trafo3w.lv_bus in net.bus_geodata.index
        ]
        if len(trafo3w_buses_with_geo_coordinates) > 0:
            tc = plot.create_trafo3w_collection(
                net, trafo3w_buses_with_geo_coordinates, color=trafo_color
            )
            collections.append(tc)

        if plot_sgens and len(net.sgen):
            sgc = plot.create_sgen_collection(
                net, size=sgen_size, orientation=orientation_sgens * np.pi
            )
            collections.append(sgc)
        if plot_mpv_sgens and len(net.sgen):
            self.mpv_concentration_rate_percent = mpv_concentration_rate_percent
            mpv_sgens, net = self.create_mpv_sgens_to_load_buses_forMPVs(net)
            if len(mpv_sgens):
                mpv_sgc = plot.create_sgen_collection(
                    net,
                    sgens=mpv_sgens,
                    size=mpv_sgen_size,
                    orientation=orientation_mpv_sgens * np.pi,
                )
                collections.append(mpv_sgc)
        if plot_loads and len(net.load):
            lc = plot.create_load_collection(
                net, size=load_size, orientation=orientation_loads * np.pi
            )
            collections.append(lc)
        if plot_storages and len(net.storage):
            stoc = plot.create_storage_collection(
                net,
                storages=net.storage.index,
                size=storage_size,
                orientation=orientation_storage * np.pi,
            )
            # stoc = plot.create_bus_collection(net,
            #                                   buses=net.storage.bus,
            #                                   size=storage_size*0.65,
            #                                   patch_type="poly5",
            #                                   color="red",
            #                                   z=None)
            collections.append(stoc)
        if plot_annotation:
            # Create annotation collection for bus indices.
            coords = zip(
                net.bus_geodata.x.loc[net.bus.index.tolist()].values + 0.10,
                net.bus_geodata.y.loc[net.bus.index.tolist()].values + 0.10,
            )
            bus_annotation = plot.create_annotation_collection(
                size=annotation_size,
                texts=np.char.mod("%d", net.bus.index.tolist()),
                coords=coords,
                zorder=0.1,
                color="black",
            )
            collections.append(bus_annotation)
        ax = plot.draw_collections(collections, ax=ax)
        plt.show()
        return ax

    def _integrate_sgens_into_storage_buses_for_PVBES(self, net):
        """
        Integriert zusätzliche PV-Anlagen bzw. statische Generatoren (sgens)
        im Stromnetz basierend auf den Speicherbussen, die keine PV-Anlagne besitzen.
        Args:
        net (pandapowerNet): Das pandapower Netzwerk, in das die sgens eingefügt werden sollen.
        Returns:
        net (pandapowerNet): Die Funktion modifiziert das Netzwerk direkt und gibt es zurück.
        """
        try:
            storage_buses = net.storage.bus
            sgen_buses = net.sgen.bus
            num_sgen = len(net.sgen)
            for i in range(len(storage_buses)):
                storage_bus_id = storage_buses[i]
                sgen_bus_id = sgen_buses[i]
                sgen_at_bus = net.sgen[net.sgen["bus"] == sgen_bus_id]
                p_mw = sgen_at_bus["p_mw"].iloc[0]
                q_mvar = sgen_at_bus["q_mvar"].iloc[0]
                sn_mva = sgen_at_bus["sn_mva"].iloc[0]
                sgen_type = sgen_at_bus["type"].iloc[0]
                voltLvl = sgen_at_bus["voltLvl"].iloc[0]
                profile = sgen_at_bus["profile"].iloc[0]
                phys_type = sgen_at_bus["phys_type"].iloc[0]
                subnet = sgen_at_bus["subnet"].iloc[0]
                sgen_bes_name = f"PV{sgen_bus_id}-BES{storage_bus_id} SGen {i+num_sgen}"
                pp.create_sgen(
                    net,
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
            return net
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

    def create_mpv_sgens_to_load_buses_forMPVs(self, net):
        """
        Erstellt statische Mini Photovoltaic(MPV) Generatoren (sgens) im Stromnetz
        basierend auf den Lastbussen.
        Args:
        net (pandapowerNet): Das pandapower Netzwerk, in das die sgens eingefügt werden sollen.
        """
        try:
            # Retrieve the list of load buses from the network
            load_buses = net.load.bus
            # Calculate the total number of load buses in the network
            total_loads = len(net.load)
            total_sgens = len(net.sgen)
            # Determine the number of load buses influenced by the mpv penetration rate
            num_mpv_influenced_load_buses = int(
                (self.mpv_concentration_rate_percent / 100) * total_loads
            )
            # Retrieve the list of static generator buses from the network
            sgen_buses = net.sgen.bus
            # Define the method for selecting bus IDs
            # Options: "random" for random selection, "sequential" for sequential selection
            selection_method = "random"  # Change to "sequential" as needed
            # selection_method = "sequential"  # Change to "sequential" as needed
            # Output configuration and calculated parameters
            width = 10
            print(f"Selection Method of MPVs:       {selection_method:>{width}s}")
            print(
                f"mpv_concentration_rate_percent: {self.mpv_concentration_rate_percent:>{width}.2f} %"
            )
            print(f"total_loads:                    {total_loads:>{width}d}")
            print(
                f"num_mpv_influenced_load_buses:  {num_mpv_influenced_load_buses:>{width}d}"
            )
            print(f"num_pv:                         {total_sgens:>{width}d}")
            print(
                f"mpv_inverter_apparent_power_mva: {self.mpv_inverter_apparent_power_mva:>{width-1}.6f} MW (gamma_1)"
            )
            print(
                f"mpv_solar_cell_capacity_mw:     {self.mpv_solar_cell_capacity_mw:>{width}.6f} MW (gamma_2)"
            )
            for i in range(num_mpv_influenced_load_buses):
                if selection_method == "random":
                    # Using the random module to select a load bus ID and sgen bus ID
                    selected_index = random.choice(load_buses.index)
                    selected_load_bus_id = load_buses[selected_index]
                    selected_sgen_bus_id = random.choice(sgen_buses)
                elif selection_method == "sequential":
                    # Using the modulo operator for the case where there are fewer sgens than loads
                    selected_index = i % len(load_buses)
                    selected_load_bus_id = load_buses.iloc[selected_index]
                    selected_sgen_bus_id = sgen_buses[i % len(sgen_buses)]
                if selected_load_bus_id in net.sgen["bus"].values:
                    filtered_sgens_name = ~net.sgen["name"].str.contains("SGenMPV")
                    sgen_at_bus = net.sgen[
                        (net.sgen["bus"] == selected_load_bus_id) & filtered_sgens_name
                    ]
                else:
                    sgen_at_bus = net.sgen[net.sgen["bus"] == selected_sgen_bus_id]
                # Entfernen Sie das ausgewählte Element aus der Liste
                load_buses = load_buses.drop(selected_index)
                sn_mva = self.mpv_inverter_apparent_power_mva
                p_mw = sn_mva * 0.99
                q_mvar = sn_mva * np.sqrt(1 - 0.99**2)
                sn_mva = np.sqrt(p_mw**2 + q_mvar**2)
                sgen_type = "MPV"
                voltLvl = sgen_at_bus["voltLvl"].iloc[0]
                profile = sgen_at_bus["profile"].iloc[0]
                phys_type = sgen_at_bus["phys_type"].iloc[0]
                subnet = sgen_at_bus["subnet"].iloc[0]
                sgen_bes_name = (
                    f"PV{selected_sgen_bus_id}-Load{selected_load_bus_id} SGenMPV {i}"
                )
                pp.create_sgen(
                    net,
                    bus=selected_load_bus_id,
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
                print(
                    f"SGenMPV {i} added to grid in Bus Id {selected_load_bus_id}. Total Bus: {len(load_buses)+1}"
                )
            # Count entries containing "SGenMPV":
            num_sgen_mpv = net.sgen.name.str.contains("SGenMPV").sum()
            # Count entries containing "SGenMPV":
            num_load = len(net.load)
            # Extract numbers from entries with "SGenMPV" in their names
            extract_sgen_mpv_num = (
                net.sgen.name[net.sgen.name.str.contains("SGenMPV")]
                .str.extract("(\d+)$")
                .astype(int)
            )
            # Determine the highest number among these entries
            max_sgen_mpv_number = extract_sgen_mpv_num.max().values[0] + 1
            # Compare the count of entries to the highest number + 1
            mpv_sgens = extract_sgen_mpv_num.index
            if num_sgen_mpv == num_load:
                print(
                    f"The comparison matches."
                    f"total_loads: {total_loads}"
                    f"num_mpv_influenced_load_buses: {num_mpv_influenced_load_buses})"
                    f"(MPV total {num_sgen_mpv})."
                )
            else:
                print("!The comparison does not match!")
            return mpv_sgens, net
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

    def cfg_plot_scenarios(self, multi_plot_params, plot_params):
        """
        Erstellt Plots für verschiedene Szenarien und Codes.

        :param multi_plot_params: Ein Wörterbuch mit Einstellungen für das Multi-Plotting.
        :param get_simbench_code: Funktion, um den Simbench-Code zu erhalten.
        :param load_simbench_powergrid: Funktion, um das Simbench-Stromnetz zu laden.
        :param advanced_plot: Funktion für das fortgeschrittene Plotten.
        :param basic_plot_params: Grundlegende Plot-Parameter.
        """
        # Speichern der Werte in einzelne Variablen
        num_codes = multi_plot_params["num_codes"]
        num_cols = multi_plot_params["num_cols"]
        num_rows = multi_plot_params["num_rows"]
        num_scenarios = multi_plot_params["num_scenarios"]

        # Iterieren durch alle Szenarien
        for scenario_index in range(num_scenarios):
            # Erstellen einer Figur für alle Plots im aktuellen Szenario
            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=(18, 6)
            )  # Größe anpassen

            # Index für den aktuellen Plot
            plot_index = 0

            # Iterieren durch alle Codes für das aktuelle Szenario
            for code_index in range(num_codes):
                # Berechnen der Subplot-Indizes
                row_index = plot_index // num_cols
                col_index = plot_index % num_cols
                ax = axes[row_index, col_index]

                # Debugging-Informationen
                print(
                    f"Plotting: scenario_index={scenario_index}, code_index={code_index}, plot_index={plot_index}, row={row_index}, col={col_index}"
                )

                # Laden des Simbench-Codes und des Netzwerks
                sb_code = self.get_simbench_code(code_index, scenario_index)
                net, simbench_code = self.load_simbench_powergrid(
                    code_index=code_index,
                    scenario_index=scenario_index,
                    rotate_plot=False,
                )
                # net = self._integrate_sgens_into_storage_buses_for_PVBES(net)

                # Entfernen der Line Geodaten
                net.line_geodata.drop(
                    set(net.line_geodata.index) & set(net.line.index), inplace=True
                )

                # Erstellen der Plots für das aktuelle Netzwerk
                self.advanced_plot(net, ax=ax, **plot_params)

                # Optional: Titel für jeden Subplot setzen
                ax.set_title(f"Code: {sb_code}")

                plot_index += 1

            plt.show()

    def automatic_plot_save_for_each_code_separate(
        self, file_path, num_codes, scenario_index, plot_params
    ):
        """
        Erstellt Plots für jeden Code innerhalb eines gegebenen Szenarios.

        :param num_codes: Anzahl der Codes, für die Plots erstellt werden sollen.
        :param scenario_index: Der Index des aktuellen Szenarios.
        :param plot_params: Grundlegende Plot-Parameter.
        """
        for code_index in range(num_codes):
            # Erstellen einer Figur für den aktuellen Plot
            fig, ax = plt.subplots(figsize=(18, 6))  # Größe anpassen

            # Laden des Simbench-Codes und des Netzwerks
            sb_code = self.get_simbench_code(code_index, scenario_index)
            net, simbench_code = self.load_simbench_powergrid(
                code_index=code_index, scenario_index=scenario_index, rotate_plot=False
            )
            # Entfernen der Line Geodaten
            net.line_geodata.drop(
                set(net.line_geodata.index) & set(net.line.index), inplace=True
            )

            # Erstellen des Plots für das aktuelle Netzwerk
            self.advanced_plot(net, ax=ax, **plot_params)

            # Optional: Titel für den Plot setzen
            # ax.set_title(f"Code: {sb_code}")
            # Layout optimieren
            plt.tight_layout()

            # Speichern des Plots als PDF
            plt.savefig(f"{file_path}plot_{sb_code}.pdf")
            # Speichern als TikZ
            # tikzplotlib.save(f"{code_index}_{sb_code}.tex")
            # Anzeigen des Plots
            plt.show()

    def plot_distribution_grid(self, net, choice, ax=None):
        """
        Plots the distribution grid based on the specified choice of plot type.
        :param net: The distribution grid object to be plotted.
        :param choice: A string indicating the type of plot - either 'advanced_plot' or 'simple_plot'.
        :param ax: Optional matplotlib axis object to plot on. If None, a new one will be created.
        :return: The matplotlib axis with the plot, or None if an invalid choice was provided.
        """
        if choice == "advanced_plot":
            # Apply advanced plotting parameters
            plot_params = self.advanced_plot_params
            ax = self.advanced_plot(net, ax=ax, **plot_params)
        elif choice == "simple_plot":
            # Apply simple plotting parameters
            plot_params = self.simple_plot_params
            ax = self.simple_plot(net, ax=ax, **plot_params)
        else:
            # Handle invalid input
            print(
                "Invalid input. Please choose either 'advanced_plot' or 'simple_plot'."
            )
            return None
        return ax

    def simple_script(self, choice="advanced_plot"):
        """ """
        ###
        net, simbench_code = self.load_simbench_powergrid(
            code_index=0, scenario_index=2, rotate_plot=False
        )
        # Add PVs to BES
        net = self._integrate_sgens_into_storage_buses_for_PVBES(net)
        # Remove Line Geodata
        net.line_geodata.drop(
            set(net.line_geodata.index) & set(net.line.index), inplace=True
        )
        fig, ax = plt.subplots(figsize=(18, 6))  # Größe anpassen
        # Advanced plotting with specific parameters
        self.plot_distribution_grid(net=net, choice=choice, ax=ax)
        return net, ax


# Main Execution Block
if __name__ == "__main__":
    # Get the base path of the current script file
    base_path = os.path.dirname(os.path.abspath("__file__"))
    yaml_path = os.path.dirname(base_path)
    obj = PlotConfigManager(base_path, yaml_path)
    obj.update_plot_configuration()

    # I. (1): simple_plot (2): advanced_plot
    net, ax = obj.simple_script(choice="advanced_plot")
    # II. Plots 6 LV Grids ((1):bus, (2): load,sgen, (3): load,sgen,storage,mpv)
    # obj.cfg_plot_scenarios(multi_plot_params=obj.multi_plot_params,
    #                        plot_params=obj.simple_plot_params)
    # III. PRINT PDF PLOTS
    # Plots based on num_codes[1,2,3,4,5] and scenario_index[0,1,2]
    # obj.automatic_plot_save_for_each_code_separate(file_path=obj.cfg_plot["path"]["save_output_plot_path"],
    #                                             num_codes=1, #len(valid_sb_code),
    #                                             scenario_index=2, #len(valid_sb_scenario),
    #                                             plot_params=obj.basic_plot_params)
    # IV. PRINT LATEX TABLE FOR PAPER
    # latex_tab345 = obj.paper_gen_latex_table(file_path=obj.cfg_plot["path"]["save_output_latex_path"],
    #                                          flag_PVBES=False)
