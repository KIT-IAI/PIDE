"""PowerGridEvaluator Module."""

import os
import datetime
import base64
import yaml
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Importing modules from subdirectories
from .helper_powergrid_advanced_plot import PlotConfigManager

# from .helper_powergrid_renderer import PowerGridRenderer


class PowerGridEvaluator:
    """
    This class evaluates the performance of a power grid by computing various
    performance metrics such as voltage stability, power flow, and transient
    stability.
    It expects a power grid environment represented by an instance of the
    parent class 'PowerGridManagerClass', which includes a pandapower net and
    other related data and functions for evaluating the power grid.
    Parameters:
    -----------
    pg_rbc_pp: PowerGridRuleBasedControlPP
        An instance of the 'PowerGridRuleBasedControlPP' class representing the power grid
        configuration and data, and the time series simulation for which the
        KPIs are to be computed.
    Background:
        The time series simulation is performed in PowerGridRuleBasedControlPP and
        BasePowerGridExtended is used as part of the time series simulation process
        (inherited within PowerGridRuleBasedControlPP).
        Therefore, an instance of "PowerGridRuleBasedControlPP" is passed to
        'PowerGridEvaluator' instead of 'BasePowerGridExtended' or
        'BasePowerGridCore'.
    Usage:
    ------
    >>> pg_rbc_pp = PowerGridRuleBasedControlPP(arguments)
    >>> pg_rbc_pp.run()
    >>> pg_rbc_pp_pge = PowerGridEvaluator(pg_rbc_pp)
    """

    def __init__(self, powergrid_env):
        """Initialize PowerGridEvaluator class with a powergrid environment.
        Parameters:
        -----------
        powergrid_env : PowerGridRuleBasedControlPP
            An instance of the 'PowerGridRuleBasedControlPP' class representing
            the power grid configuration and data, and the time series simulation
            for which the KPIs are to be computed.
        """
        # Assign power grid environment(object) and network model to this class instance
        self.powergrid_env = powergrid_env
        self.net = self.powergrid_env.net

        # Load the configuration from the YAML file
        with open(self.powergrid_env.cfg_user_plot_path, "r") as file:
            self.cfg_plot = yaml.safe_load(file)

        # Handling Monte Carlo simulation results
        if powergrid_env.flag_monte_carlo:
            # Validate variability in simulation results
            self.check_difference_in_mcs_results()
            # Initialize dictionary to store aggregated simulation data
            mcs_log_variables = {}
            keys = self.raw_log_mcs_data[0].output.keys()  # Variable names
            num_dicts = len(self.raw_log_mcs_data)  # Number of simulation datasets
            # Aggregate and average data for each variable
            for key in keys:
                total = sum(d.output[key] for d in self.raw_log_mcs_data)
                mcs_log_variables[key] = total / num_dicts
            # Update environment's log with aggregated data
            self.powergrid_env.log_variables = mcs_log_variables
            self.keys_list = list(mcs_log_variables.keys())
            # Update path name
            # Path of the existing folder
            existing_folder = self.powergrid_env.output_data_path
            # Generate the new folder path
            new_folder = existing_folder + "_mcs"
            # Rename the existing folder to the new name
            os.rename(existing_folder, new_folder)
            # Update the path in the powergrid_env instance
            self.powergrid_env.output_data_path = new_folder
        else:
            # Use existing log variables if not in Monte Carlo mode
            self.keys_list = list(self.powergrid_env.log_variables.keys())

        # Set parameters of interest for analysis and evaluation
        # A flag variable for storages
        self.powergrid_env.is_storage_scenario = not self.net.storage.empty
        # time option to use datetime format for evaluation in Figures
        self.time_option = "datetime"
        # If True, include all buses. If False, only include low-voltage buses in voltage limits
        self.include_high_voltage = False
        # Activate critical limits in plots
        self.plot_activate_critic_limits = False
        # Initialize dictionary to store parameters for each network element
        self.params = {}
        self.bus_lv_mask = None
        self.buses_vm_ts_lv = None
        self.vm_limits = None
        self.vm_scalar_limits = None
        # Set the figure name based on the simulation name or raw data network name
        if len(self.powergrid_env.sb_code) > 1:
            self.figname = self.powergrid_env.sb_code
        else:
            self.figname = self.powergrid_env.args["rawdata"]["net_name"]
        # Default_nominal_voltage (default=1.00):
        self.default_nominal_voltage = 1.00
        self.vm_num_violations_damage = None
        self.vm_num_violations_critic = None
        self.transformer_num_violations_dam = None
        self.lines_num_violations_dam = None
        self.pg_renderer = None
        # Read and increment the sequence_id from the file
        self.sequence_id = self.read_and_increment_sequence_id(
            self.powergrid_env.sequence_id_path
        )
        # Define kpi dictionaries
        self.kpi_raw = {}
        self.kpi_vm_violations = {}
        self.kpi_transformer_and_lines_violations = {}
        self.kpi_power_losses_and_power_balance = {}
        self.kpi_storage = {}
        self.sim_arguments = {}
        self.kpi_key = {}
        # Figure and output file paramters:
        self.use_html = False
        self.text_output_formats = ".xlsx"  # ['csv', 'xlsx']
        self.output_file_types = [".xlsx", ".html"]
        self.output_formats = "jpeg"  # ['pdf', 'png', 'jpeg']
        self.dpi = 500
        self.plot_names = [
            "00_res_powergrid_topology_printer",
            "01_res_ts_load_plot",
            "02_res_ts_pv_plot",
            "03_res_ts_buses_plot",
            "04_res_ts_ext_grid_plot",
            "05_res_ts_lines_loss_and_loading_plot",
            "06_res_ts_transformer_loading_powerloss_plot",
            "07_res_ts_storages_p_mw_q_mvar_soc_plot",
            "08_res_submain_data_profile_plot",
            "09_res_analyze_voltage_magnitudes",
            "10_res_bus_voltage_magnitudes_subplots",
            "11_res_lines_and_transformer_overloading",
        ]
        self.file_name = [f"log_00_res_kpi_sequence_id_{self.sequence_id}"]
        # Voltage criticality criteria
        self.v_nom = None
        self.v_nom_net = None
        self.v_min = None
        self.v_max = None
        self.v_min_max_delta = None
        self.v_crit_lower = None
        self.v_crit_upper = None
        self.transformer_max = None  # TransformerOverheating
        self.lines_max = None  # GridLineCongestion
        # DER Settings:

        # Regulation standard for DER systems
        self.regulation_standard = self.powergrid_env.regulation_standard
        # Base Inverter Params
        self.base_inverter_ctrl_params = self.powergrid_env.inverter_ctrl_params
        # Inverter control mode parameters
        self.inverter_ctrl_params = self.base_inverter_ctrl_params[
            self.regulation_standard
        ]
        # Regulation standard mode parameters
        self.regulation_standard_mode = self.inverter_ctrl_params["standard_mode"]
        # Selected Inverter Control and Criticality parameters
        self.selected_inverter_ctrl_params = self.inverter_ctrl_params[
            self.regulation_standard_mode
        ]
        self.criticality_params = self.base_inverter_ctrl_params["criticality"]

        self.load_ctrl_mode_and_technical_params()
        # Start Time-Series Evaluation
        self._run_evaluation()
        # Start Time-Series Evaluation

    def _run_evaluation(self):
        """PowerGridEvaluator Time Series Evaluation"""
        print("\n----------PowerGridEvaluator Time Series Evaluation-----------\n")
        # (1) "Visualization of the power grid toplogy".
        self.res_powergrid_topology_printer()
        # (2) "Rigorous examination of individual log variables"(*)
        self.res_timeseries_plot()
        # (3) "Critical and Rigorous Evaluation"(*)
        self.res_submain_data_profile()
        # (4) "Key Performance Indicators (KPIs) for a power grid"(*)
        self.res_submain_key_performances()
        print("\n------------------- Successfully executed --------------------\n")

    def read_and_increment_sequence_id(self, file_path):
        """Read and Increment Sequence ID
        Read the sequence ID from the specified file, increment it, and save
        the updated sequence ID back to the file.
        Parameters:
        file_path (str): The path of the file containing the sequence ID.
        Returns:
        int: The current sequence ID before being incremented.
        """
        # Open the file in read mode
        with open(file_path, "r", encoding="utf-8") as file:
            # Read file, remove any leading/trailing whitespace, and convert it to an integer
            sequence_id = int(file.read().strip())
        # Open the file in write mode
        with open(file_path, "w", encoding="utf-8") as file:
            # Increment the sequence_id and write it back to the file
            file.write(str(sequence_id + 1))
        # Print the current sequence_id to the console
        print(
            "\n---------------- Sequence ID -----------------\n"
            f"Sequence-ID: {sequence_id} \n"
        )
        return sequence_id

    def save_and_display_figure(self, fig_name, plot_file_name):
        """Saves a matplotlib figure to disk in the specified self.output_formats,
        and optionally shows the figure.
        Parameters:
        -----------
        fig_name : matplotlib Figure object
            The figure to be saved and shown.
        plot_file_name : str
            The file name (without extension) to be used when saving the figure.
        self.output_formats : list of str
            The list of file formats to save the figure in (e.g. ['pdf', 'png', 'jpeg']).
        self.dpi : int
            The resolution (dots per inch) to be used when saving the figure.
        """
        plt.tight_layout()
        # Iterate over the specified output formats
        for output_format in [self.output_formats]:
            file_name = f"{plot_file_name}.{output_format}"
            # Construct the output file path
            output_file = os.path.join(self.powergrid_env.output_data_path, file_name)
            # Save the figure in the current format
            fig_name.savefig(output_file, dpi=self.dpi)
        # Convert the figure to HTML
        if self.use_html is True:
            file_name = f"{plot_file_name}.html"
            # Construct the HTML file path
            output_file = os.path.join(self.powergrid_env.output_data_path, file_name)
            self.write_fig2html(fig_name, file_name_html=output_file)
        # Display the plot
        plt.show()
        # Close all figure windows
        plt.close("all")
        print(f"{plot_file_name}.{output_format}")

    def handle_additional_output_file_types(self, file_name_base, dict_data):
        """This function handles the saving of additional file formats like
        .html and .xlsx. It takes in a base file name and a dictionary of data
        to save.
        """
        for output_file_type in self.output_file_types:
            # Construct the output file path
            output_file = os.path.join(
                self.powergrid_env.output_data_path, file_name_base + output_file_type
            )
            print(f"'{output_file}'")
            if output_file_type == ".html":
                # Open the file in write mode
                with open(output_file, "w", encoding="utf-8") as file:
                    for name, data_frame in dict_data.items():
                        # Write a header for each DataFrame
                        file.write(f"<h1>{name}</h1>")
                        # Write the DataFrame as HTML
                        file.write(data_frame.to_html())
            if output_file_type == ".xlsx":
                # Create an ExcelWriter object
                with pd.ExcelWriter(output_file) as writer:
                    for name, data_frame in dict_data.items():
                        # Write the DataFrame to an Excel sheet
                        data_frame.to_excel(writer, sheet_name=name)

    @staticmethod
    def write_fig2html(
        fig, file_name_html="datasets/plot_save/00_res_timeseries_load_plot.html"
    ):
        """
        Save a Matplotlib figure object as an HTML file with an embedded PNG
        image.
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The figure object to be saved as an image and embedded in the HTML file.
        file_name_html : str, optional
            The name of the HTML file to be created.
            Defaults to "datasets/plot_save/00_res_timeseries_load_plot.html".
        Returns:
        --------
            None
        Notes:
        ------
        The method saves the figure object as a PNG image in memory and encodes
        it using base64.The encoded image is then embedded in an HTML template
        string, which includes some boilerplate HTML code before and after the
        image tag. Finally, the resulting HTML string is written to a file with
        the specified file_name_html.
        Example:
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [4, 5, 6])
        >>> write_fig2html(fig, "datasets/plot_save/myplot.html")
        """
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format="png")
        encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
        html = (
            f"Some html head <img src='data:image/png;base64,{encoded}'> Some more html"
        )
        # html = 'Some html head' + \
        #     '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + \
        #     'Some more html'
        with open(file_name_html, "w", encoding="utf-8") as file:
            file.write(html)

    def check_difference_in_mcs_results(self):
        """Checks the difference between two sets of Monte Carlo simulation results.
        Specifically compares 'sgen.p_mw' values from two different simulations.
        Prints a success message if the difference is not zero, indicating variability.
        """
        # Set raw monte carlo simulation data
        self.raw_log_mcs_data = self.powergrid_env.raw_log_mcs_data
        # Retrieve raw Monte Carlo simulation data
        self.raw_log_mcs_data_0 = self.raw_log_mcs_data[0].output["sgen.p_mw"]
        self.raw_log_mcs_data_1 = self.raw_log_mcs_data[1].output["sgen.p_mw"]
        # Calculate the difference between the two sets of data
        difference = self.raw_log_mcs_data_0 - self.raw_log_mcs_data_1
        # Check if the difference is not zero
        if sum(difference) != 0:
            print("Test successful for pseudo-random generator with seed value.")
        # self.global_seed_value

    def res_powergrid_topology_printer(self):
        """Renders the powergrid topology using the PowerGridRenderer, adjusts
        the layout to avoid overlap, saves and displays the resulting figure."""
        # Access various plot plot_configuration parameters
        advanced_plot_params = self.cfg_plot[
            "main_advanced_plot"
        ]  # 6 LV Grids for every scenario (load,sgen,storage)
        figure, ax = plt.subplots(figsize=(18, 6))  # Größe anpassen
        advanced_plot_params["mpv_concentration_rate_percent"] = (
            self.powergrid_env.mpv_concentration_rate_percent
        )
        plot_params = advanced_plot_params
        obj = PlotConfigManager(
            self.powergrid_env.helper_path, self.powergrid_env.yaml_path
        )
        ax = obj.advanced_plot(net=self.net, ax=ax, **plot_params)
        self.save_and_display_figure(fig_name=figure, plot_file_name=self.plot_names[0])

    # def res_powergrid_renderer_topology_plot(self):
    #     """ Renders the powergrid topology using the PowerGridRenderer, adjusts
    #     the layout to avoid overlap, saves and displays the resulting figure. """
    #     # Importing modules from subdirectories
    #     from .helper_powergrid_renderer import PowerGridRenderer
    #     self.pg_renderer = PowerGridRenderer(net=self.powergrid_env.net,
    #                                          environment=self.powergrid_env.net,
    #                                          simbench_code=self.powergrid_env.sb_code,
    #                                          option="imported_module_plot_mode_1")
    #     figure, _ = self.pg_renderer.main_render()
    #     self.save_and_display_figure(fig_name=figure,
    #                                  plot_file_name=self.plot_names[0])

    def res_timeseries_plot(self):
        """
        Creates three plots based on the key_list which contains the saved data
        of log_variables, including
        res_load:{p_mw, q_mvar}, sgen:{p_mw, q_mvar},
        res_bus:{p_mw, q_mvar, vm_pu, va_degree},
        res_line:{pl_mw, loading_percent},
        res_trafo:{p_hv_mw, loading_percent}, and
        res_ext_grid:{p_mw, q_mvar}.
        It initializes subplots and sets titles, labels, and legends for each
        axis. The function also maximizes the window of the plot.
        """
        spec_label = self.res_timeseries_preparation()
        self.plot_power_grid_elements(spec_label)

    def plot_power_grid_elements(self, spec_label):
        """Plotting Powergrid Elements results bus,lines ...
        self.figname
        self.powergrid_env.scenario
        self.powergrid_env.mcs_log_variables
        self.keys_list
        self.save_and_display_figure
        self.plot_names
        """
        # Define the base shapes for subplots. Each tuple represents a subplot layout.
        base_subplot_shapes = [(1, 2), (1, 2), (2, 2), (1, 2), (1, 3), (1, 5)]
        # Extend the subplot shapes with an additional shape if it's a storage scenario.
        if self.powergrid_env.is_storage_scenario:
            base_subplot_shapes.append((1, 4))
        if self.powergrid_env.mpv_flag:
            base_subplot_shapes.insert(2, (1, 2))
            self.plot_names.insert(2, "02_res_ts_mpv_plot")  # 3
            self.keys_list.insert(2, "sgen.p_mw")
            self.keys_list.insert(3, "sgen.q_mvar")
        # Initialize the list to store cumulative subplot counts.
        total_subplots_per_shape = [0]  # Start value is 0.
        # Calculate cumulative number of subplots for each shape.
        for shape in base_subplot_shapes:
            # Multiply the dimensions of the shape to get the number of subplots.
            num_subplots = shape[0] * shape[1]
            # Add to the cumulative count.
            total_subplots_per_shape.append(total_subplots_per_shape[-1] + num_subplots)
        # Create a list of ranges, each representing the subplot indexes for each shape.
        total_element_range = [
            range(total_subplots_per_shape[i], total_subplots_per_shape[i + 1])
            for i in range(len(total_subplots_per_shape) - 1)
        ]
        z = 0
        # Schleife durch subplot_shapes und element_range gleichzeitig
        for k_i, (shape, element_idx) in enumerate(
            zip(base_subplot_shapes, total_element_range), start=1
        ):
            fig, axes = plt.subplots(shape[0], shape[1], figsize=(18.5, 10.5))
            axes = (
                axes.flatten()
            )  # Flatten the array of axes, so we can iterate over it
            fig.suptitle(
                f'Analysis/Insights "{self.figname}" scenario "{self.powergrid_env.scenario}"',
                fontsize=12,
            )
            for iter_axex, iter_i in zip(axes, element_idx):
                iter_axex.set_title(f"{self.keys_list[iter_i]}")
                print(f"{self.keys_list[iter_i]}")
                plot_data = self.powergrid_env.log_variables[self.keys_list[iter_i]]
                if (
                    self.keys_list[iter_i] == "sgen.p_mw"
                    or self.keys_list[iter_i] == "sgen.q_mvar"
                ):
                    if self.powergrid_env.mpv_flag and z < 2:
                        # range_index = self.powergrid_env.mpv_range_index
                        # plot_data = self.powergrid_env.log_variables[self.keys_list[iter_i]]
                        # plot_data = plot_data[range_index]
                        mask_mpv = self.net.sgen.type == "MPV"
                        plot_data = plot_data.loc[:, mask_mpv]
                        z += 1
                        if z >= 2:
                            self.mpv_flag = False
                    if z > 2:
                        plot_data = self.powergrid_env.log_variables[
                            self.keys_list[iter_i]
                        ]
                        range_index = self.powergrid_env.pv_range_index
                        plot_data = plot_data[range_index]
                        mask_pv = self.net.sgen.type == "PV"
                        plot_data = plot_data.loc[:, mask_pv]
                line_objects = iter_axex.plot(plot_data)
                iter_axex.set_xlabel("timesteps")
                iter_axex.set_ylabel("Y axis")
                iter_axex.legend(line_objects, (spec_label[iter_i]))
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig_manager = plt.get_current_fig_manager()
            fig.set_size_inches(18.5, 10.5, forward=False)
            self.save_and_display_figure(
                fig_name=fig, plot_file_name=self.plot_names[k_i]
            )

    def res_timeseries_preparation(self):
        """res_timeseries_preparation"""
        load_no, loadbus_no = self.net.load.index.to_list(), self.net.load.bus.to_list()
        sgen_no, sgenbus_no = self.net.sgen.index.to_list(), self.net.sgen.bus.to_list()
        bus_no = self.net.bus.index.to_list()
        # Initialise the subplot function using number of rows and columns
        load_label = [
            f"load{load_no[i]} bus{loadbus_no[i]}" for i in range(len(self.net.load))
        ]
        sgen_label = [
            f"{self.net.sgen.type[i]}{sgen_no[i]} bus{sgenbus_no[i]}"
            for i in range(len(self.net.sgen))
        ]
        bus_label = [f"bus{bus_no[i]}" for i in range(len(self.net.bus))]
        line_labels = [
            f"{line_name} from_bus{from_bus}_to_bus{to_bus}"
            for line_name, from_bus, to_bus in zip(
                self.powergrid_env.net.line.name.values,
                self.powergrid_env.net.line.from_bus.values,
                self.powergrid_env.net.line.to_bus.values,
            )
        ]
        trafo_labels = [
            f"trafo hv_bus{hv_bus}_lv_bus{lv_bus}"
            for hv_bus, lv_bus in zip(  # self.powergrid_env.net.trafo.name.values,
                self.powergrid_env.net.trafo.hv_bus.values,
                self.powergrid_env.net.trafo.lv_bus.values,
            )
        ]
        ext_grid_labels = [
            f"{ext_grid_name} bus{busnr}"
            for ext_grid_name, busnr in zip(
                self.powergrid_env.net.ext_grid.name.values,
                self.powergrid_env.net.ext_grid.bus.values,
            )
        ]
        # Combine spec_labels into a dictionary
        spec_label_base = [
            load_label,
            load_label,
            sgen_label,
            sgen_label,
            bus_label,
            bus_label,
            bus_label,
            bus_label,
            ext_grid_labels,
            ext_grid_labels,
            line_labels,
            line_labels,
            line_labels,
            trafo_labels,
            trafo_labels,
            trafo_labels,
            trafo_labels,
            trafo_labels,
        ]
        if self.powergrid_env.mpv_flag:
            mpv_sgen_label = [
                f"{self.net.sgen.type[i]}{sgen_no[i]} bus{sgenbus_no[i]}"
                for i in self.powergrid_env.mpv_range_index
            ]
            spec_label_base.insert(2, mpv_sgen_label)  # First insertion
            spec_label_base.insert(3, mpv_sgen_label)  # Second insertion
        if self.powergrid_env.is_storage_scenario:
            storage_bus_no = self.powergrid_env.net.storage.bus.to_list()
            storage_no = self.powergrid_env.net.storage.index.to_list()
            storage_label = [
                f"storage{storage_no[i]} bus{storage_bus_no[i]}"
                for i in range(len(self.powergrid_env.net.storage))
            ]
            spec_label_base.extend(
                [storage_label, storage_label, storage_label, storage_label]
            )
        return spec_label_base

    def res_submain_data_profile(self):
        """Generate a plot showing the mean and confidence intervals of active
        power for each network element.
        self.net_elements = ["load", "sgen", "storage"]"""
        # Create a figure and axes object
        fig9, axis = plt.subplots(figsize=(10, 5))

        # Loop through all network elements
        for element in self.powergrid_env.net_elements:
            # Analyze data for the current element and store results in self.params
            self._submain_analyze_data(element)
            self.plot_element_data_and_confidence_intervals(element, axis)
        # Set the x-axis and y-axis labels and add legend
        axis.set_xlabel(self.params["load"]["xlabel"] or "Timesteps")
        axis.set_ylabel("Active Power [MW]")
        axis.legend()
        # Format the x-axis to show only even hour values
        axis.xaxis.set_major_locator(
            plt.matplotlib.dates.HourLocator(byhour=range(0, 24), interval=2)
        )
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H\n"))
        if self.powergrid_env.mpv_flag:
            plot_file_name = self.plot_names[9]
        else:
            plot_file_name = self.plot_names[8]
        # Save the plot as a JPEG file and display it
        self.save_and_display_figure(fig_name=fig9, plot_file_name=plot_file_name)

    def plot_element_data_and_confidence_intervals(self, element, axis):
        """Plots the data for a specific element and its confidence intervals on
        the given axis. This function plots the mean of the data for a given element
        and shades the region between various confidence intervals (90%, 95%, and 99%).
        The parameters such as color and transparency (alpha) of the plot are defined
        in the `params` attribute of the instance of this class.
        Parameters:
        element (str):
            The key to look up the element parameters and data in `self.params`.
        axis (matplotlib.axis):
            The axis object on which to plot the data.
        Returns:
            None
        """
        axis.plot(
            self.params[element]["timesteps"],
            self.params[element]["df_mean"],
            label=self.params[element]["txt_label"],
            color=self.params[element]["color"],
        )
        ci_levels = [
            "ci_lower90",
            "ci_upper90",
            "ci_lower95",
            "ci_upper95",
            "ci_lower99",
            "ci_upper99",
        ]
        for ci_lower, ci_upper in zip(ci_levels[::2], ci_levels[1::2]):
            axis.fill_between(
                self.params[element]["timesteps"],
                self.params[element][ci_lower],
                self.params[element][ci_upper],
                facecolor=self.params[element]["color"],
                alpha=self.params[element]["alpha"],
            )

    def _submain_analyze_data(self, element):
        """
        The function takes in data and self.powergrid_env.timesteps_per_day
        as input. It calls the _calculate_mean_dataframe function to calculate the
        mean of the data for each time step.
        It then calculates the mean and standard deviation of the resulting DataFrame.
        It creates a zone mapping variable using a zone Series, and a dictionary
        that maps the indexes of the zone Series to the zone names.
        It then calls the _calculate_zone_dataframe function to calculate the mean
        of each zone and returns a new DataFrame.
        It then calls the _calculate_confidence_interval function three times with
        different alpha values to calculate the lower and upper bounds of the 99%,
        95%, and 90% confidence intervals.
        It stores all the calculated values in a dictionary with the keys as
        the names of the values and returns it.
        """
        if element == "load":
            # "load"
            load_data = self.powergrid_env.log_variables[
                "res_load.p_mw"
            ]  # load_active [MW]
            data = pd.DataFrame(load_data) if isinstance(load_data, list) else load_data
            zone_series = self.powergrid_env.net.load.zone
            element_infos = {
                "element": element,
                "txt_label": "Load Consumption",
                "xlabel": "Timesteps",
                "ylabel": f"{element.capitalize()} Active Power [MW]",
                "color": "orange",
                "alpha": 0.2,
            }
            assert_message = f"Data shape mismatch, expected {len(self.net.load)} \
                columns but got {data.shape[1]} for element {element}. \
                Verify data compatibility and check for any missing columns."
            assert data.shape[1] == len(self.powergrid_env.net.load), assert_message
        elif element == "sgen":
            # "sgen"
            sgen_data = self.powergrid_env.log_variables["sgen.p_mw"]  # pv_active [MW]
            data = pd.DataFrame(sgen_data) if isinstance(sgen_data, list) else sgen_data
            zone_series = self.powergrid_env.net.sgen.zone
            element_infos = {
                "element": element,
                "txt_label": "PV Generation",
                "xlabel": "Timesteps",
                "ylabel": f"{element.capitalize()} Active Power [MW]",
                "color": "lightskyblue",
                "alpha": 0.2,
            }
            assert_message = f"Data shape mismatch, expected {len(self.net.sgen)} \
                columns but got {data.shape[1]} for element {element}. \
                    Verify data compatibility and check for any missing columns."
            assert data.shape[1] == len(self.powergrid_env.net.sgen), assert_message
        elif element == "storage":
            # "storage"
            storage_data = self.powergrid_env.log_variables[
                "storage.p_mw"
            ]  # storage_active
            data = (
                pd.DataFrame(storage_data)
                if isinstance(storage_data, list)
                else storage_data
            )
            zone_series = self.powergrid_env.net.storage.zone
            element_infos = {
                "element": element,
                "txt_label": "Storage Profile",
                "xlabel": "Timesteps",
                "ylabel": f"{element.capitalize()} Active Power [MW]",
                "color": "green",
                "alpha": 0.2,
            }
            assert_message = f"Data shape mismatch, expected {len(self.net.storage)}\
                columns but got {data.shape[1]} for element {element}. \
                    Verify data compatibility and check for any missing columns."
            assert data.shape[1] == len(self.powergrid_env.net.storage), assert_message
        data_frame = self._calculate_mean_dataframe(data)
        data_frame_mean = data_frame[data_frame.columns].mean(axis=1)
        data_frame_std = data_frame[data_frame.columns].std(axis=1)
        # Erstellen Sie eine zone mapping-Variable aus der zone Serie
        mappings = dict(zip(zone_series.index, zone_series))
        data_frame_zone = self._calculate_zone_dataframe(data_frame, mappings)
        ci_lower99, ci_upper99 = self._calculate_confidence_interval(
            data_frame, alpha=0.01
        )
        ci_lower95, ci_upper95 = self._calculate_confidence_interval(
            data_frame, alpha=0.05
        )
        ci_lower90, ci_upper90 = self._calculate_confidence_interval(
            data_frame, alpha=0.10
        )

        if self.time_option == "datetime":
            # Create TimeSteps datetime axis
            customdate = datetime.datetime(
                2023, 1, 22, 00, 00
            )  # datetime.datetime.now()
            timesteps = np.array(
                [
                    customdate
                    + datetime.timedelta(minutes=self.powergrid_env.resolution * i)
                    for i in range(self.powergrid_env.timesteps_per_day)
                ]
            )
        else:
            # Create TimeSteps numbered axis
            timesteps = range(len(data_frame_mean))
            timesteps = range(self.powergrid_env.timesteps_per_day)

        self.params[element] = {
            "data": data,
            "timesteps_per_day": self.powergrid_env.timesteps_per_day,
            "timesteps": timesteps,
            "df": data_frame,
            "df_mean": data_frame_mean,
            "df_std": data_frame_std,
            "df_zone": data_frame_zone,
            "ci_lower99": ci_lower99,
            "ci_upper99": ci_upper99,
            "ci_lower95": ci_lower95,
            "ci_upper95": ci_upper95,
            "ci_lower90": ci_lower90,
            "ci_upper90": ci_upper90,
            "zone_series": zone_series,
            "zone_mappings": mappings,
        }
        self.params[element].update(element_infos)

    def _calculate_mean_dataframe(self, data):
        """
        # Function to calculate the mean of a DataFrame for every "step" rows
        The _calculate_mean_dataframe function takes the data and returns a new
        DataFrame with the mean values of the columns calculated every "step" rows.
        """
        data_frame = pd.DataFrame(data)
        mean_values = []
        for i in range(self.powergrid_env.timesteps_per_day):
            mean_values.append(
                [
                    data_frame[col][i :: self.powergrid_env.timesteps_per_day].mean()
                    for col in data_frame.columns
                ]
            )
        data_frame_mean = pd.DataFrame(mean_values, columns=data_frame.columns)
        return data_frame_mean

    def _calculate_zone_dataframe(self, data_frame, mappings):
        """
        The _calculate_zone_dataframe function takes a DataFrame, a mappings dictionary,
        and self.powergrid_env.timesteps_per_day as input. It creates an empty dictionary for each
        unique zone specified in the mappings dictionary.
            It then loops over all columns in the DataFrame, getting the zone for
        each column from the mappings dictionary, and calculates the mean of the
        values for each time step in the column.
            It then calls the _calculate_mean_dataframe function to calculate the
        mean of the mean values for each zone.
            Finally, it sorts the resulting DataFrame by the column names.
        """
        zones = set(mappings.values())  # Get a set of all unique zones
        zone_mean = {
            zone: {} for zone in zones
        }  # Create empty dictionary for each zone
        # Loop over all columns
        for col in data_frame.columns:
            zone = mappings[int(col)]
            mean_values = []
            # Loop over all rows
            for i in range(self.powergrid_env.timesteps_per_day):
                mean_values.append(
                    data_frame[col][i :: self.powergrid_env.timesteps_per_day].mean()
                )
            zone_mean[zone][col] = mean_values
        df_zone = pd.DataFrame()
        for zone in zone_mean:
            df_zone_temp = self._calculate_mean_dataframe(zone_mean[zone])
            df_zone[zone] = df_zone_temp[df_zone_temp.columns].mean(axis=1)
        df_zone = df_zone.sort_index(axis=1, level=1)
        return df_zone

    def _calculate_confidence_interval(self, data_frame, alpha=0.05):
        """
        The _calculate_confidence_interval function takes a DataFrame,
        Number of observations: self.powergrid_env.timesteps_per_day,
        and alpha as input and returns the lower
        and upper bounds of the confidence interval for the mean of the data.
        It uses the t-distribution to calculate the margin of error and then
        subtracts and adds it from the mean to get the lower and upper bounds
        respectively.
        """
        mean = data_frame[data_frame.columns].mean(axis=1)
        std = data_frame[data_frame.columns].std(axis=1)
        dof = self.powergrid_env.timesteps_per_day - 1  # Degrees of freedom
        t_critical = stats.t.ppf(1 - alpha / 2, dof)
        # Berechne das Konfidenzintervall
        margin_of_error = (
            t_critical * std / np.sqrt(self.powergrid_env.timesteps_per_day)
        )
        # Konfidenzintervall untere und obere Grenze
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        return ci_lower, ci_upper

    def res_submain_key_performances(self):
        """
        This function calculates various Key Performance Indicators (KPIs) for
        a power network and returns them.
        The computed KPIs include:
            KPI1: Number of voltage magnitude(vm) limit violations(vio) damage(dam)/criticality(cri)
            KPI2: Number of trafo and line loading limit violations
            KPI4: Total losses
            KPI5: Total losses in the lines
            KPI6: Average transformer loading
            KPI7: Average bus voltage magnitude
        INPUT:
            net (pandapowerNet) - A pandapower network for which the KPIs are to be computed.
        OUTPUT:
            kpis (dict) - A dictionary containing the computed KPIs.
        """
        self.res_key_performances_index_preparation()
        # KPI1: Voltage Violations
        self._calculate_vm_violations_kpi1()
        # KPI2: Transformer and Lines Violations
        self._calculate_trafo_and_lines_violations_kpi2()
        # KPI3: Power Losses and Q-Losses
        self._calculate_power_losses_and_power_balance_kpi3()
        # KPI4: Storage Self discharge
        if self.powergrid_env.is_storage_scenario:
            self._calculate_storage_utilization_kpi4()
        # Assign KPI1 values
        self.assign_kpi1()
        # Evaluation Key Performance Indexes: (1),(2) and (3)
        if self.powergrid_env.mpv_flag:
            self.plot_vm_min_max_buses_kpi1(plot_name=self.plot_names[10])
            self.plot_vm_subplots_kpi1(plot_name=self.plot_names[11])
            self.plot_lines_and_transformer_overloading(plot_name=self.plot_names[12])
        else:
            self.plot_vm_min_max_buses_kpi1(plot_name=self.plot_names[9])
            self.plot_vm_subplots_kpi1(plot_name=self.plot_names[10])
            self.plot_lines_and_transformer_overloading(plot_name=self.plot_names[11])

    def res_key_performances_index_preparation(self):
        """res_key_performances_index_preparation"""
        # Calculate the voltage limits for KPI1.
        self.vm_limits, self.vm_scalar_limits = self.compute_voltage_limits_kpi1()
        if self.include_high_voltage:
            self.bus_lv_mask = [True for _ in self.powergrid_env.lv_bus_mask]
        else:
            self.bus_lv_mask = self.powergrid_env.lv_bus_mask
        res_bus_vm = self.powergrid_env.output["res_bus.vm_pu"]
        self.buses_vm_ts_lv = res_bus_vm.loc[:, self.bus_lv_mask.values]
        bus_vm_mv = self.powergrid_env.log_variables[
            "res_bus.vm_pu"
        ]  # bus_voltage [p.u]
        bus_vm_lv = self.buses_vm_ts_lv.dropna(axis=1)
        assert (
            bus_vm_mv.shape[1] - 1 == bus_vm_lv.shape[1]
        ), "Da nur ein Trafo vorliegt -1"
        self.kpi_raw = {
            "buses_vm_ts_lv": res_bus_vm.loc[:, self.bus_lv_mask.values],
            "lines_ts_data": self.powergrid_env.output["res_line.loading_percent"],
            "trafo_ts_data": self.powergrid_env.output["res_trafo.loading_percent"],
        }
        bus_vm_stats = self.calculate_ts_data_stats_kpi1(self.kpi_raw["buses_vm_ts_lv"])
        bus_vm_ts_stats = self.get_ts_statistics(
            data_frame=self.kpi_raw["buses_vm_ts_lv"]
        )
        bus_vm_scalar_stats = self.get_scalar_vr_statistics(
            self.kpi_raw["buses_vm_ts_lv"]
        )
        lines_vm_stats = self.calculate_ts_data_stats_kpi1(
            self.kpi_raw["lines_ts_data"]
        )
        lines_ts_stats = self.get_ts_statistics(
            data_frame=self.kpi_raw["lines_ts_data"]
        )
        trafo_stats = self.calculate_ts_data_stats_kpi1(self.kpi_raw["trafo_ts_data"])
        kpi_extended = {
            "buses_vm_ts_nodes_statistics": bus_vm_stats,
            "buses_vm_ts_max_min_avg_bus_node_statistics": bus_vm_ts_stats,
            "buses_vm_ts_scalar_statistics": bus_vm_scalar_stats,
            "lines_vm_ts_edges_statistics": lines_vm_stats,
            "lines_ts_max_min_avg_edge_statistics": lines_ts_stats,
            "trafo_node_statistics": trafo_stats,
        }
        self.kpi_raw.update(kpi_extended)

    def _calculate_vm_violations_kpi1(self):
        """Counts voltage magnitude (vm) violations for 'damage' and 'critic'
        conditions in the power grid."""
        for key in ["damage", "critic"]:
            self.buses_vm_ts_lv, min_vm_pu = self.buses_vm_ts_lv.align(
                self.vm_limits["min_" + key + "_vm_pu"], axis=1, copy=False
            )
            self.buses_vm_ts_lv, max_vm_pu = self.buses_vm_ts_lv.align(
                self.vm_limits["max_" + key + "_vm_pu"], axis=1, copy=False
            )
            violations = (self.buses_vm_ts_lv < min_vm_pu) | (
                self.buses_vm_ts_lv > max_vm_pu
            )
            setattr(self, f"vm_num_violations_{key}", violations.sum().sum())

        self.kpi_vm_violations = {
            "vm_lv_mean_avg": self.kpi_raw[
                "buses_vm_ts_max_min_avg_bus_node_statistics"
            ]["average"].mean(),
            "vm_lv_mean_min": self.kpi_raw[
                "buses_vm_ts_max_min_avg_bus_node_statistics"
            ]["average"].min(),
            "vm_lv_mean_max": self.kpi_raw[
                "buses_vm_ts_max_min_avg_bus_node_statistics"
            ]["average"].max(),
            "vm_lv_mean_std": self.kpi_raw[
                "buses_vm_ts_max_min_avg_bus_node_statistics"
            ]["average"].std(),
            "vm_lv_highest_max": self.kpi_raw[
                "buses_vm_ts_max_min_avg_bus_node_statistics"
            ]["maximum"].max(),
            "vm_lv_lowest_min": self.kpi_raw[
                "buses_vm_ts_max_min_avg_bus_node_statistics"
            ]["minimum"].min(),
            "vm_num_violations_dam": f"({self.vm_num_violations_damage},"
            f"min:{self.vm_limits['min_damage_vm_pu'][1]},"
            f"max:{self.vm_limits['max_damage_vm_pu'][1]})",
            "vm_num_violations_cri": f"({self.vm_num_violations_critic},"
            f"min:{self.vm_limits['min_critic_vm_pu'][1]},"
            f"max:{self.vm_limits['max_critic_vm_pu'][1]})",
        }

    def _calculate_trafo_and_lines_violations_kpi2(self):
        """Calculate the number of violations of transformers and power lines.
        This method evaluates the status of transformers and power lines by
        comparing the loading percent of each with the critical damage threshold.
        Violations are counted when the loading percent exceeds the critical
        damage threshold.
        Attributes updated:
        self.num_trafo_violations_dam:
            The number of times the transformer's loading percent exceeds its
            critical damage threshold.
        self.num_lines_violations_dam:
            The number of times the power line's loading percent exceeds its
            critical damage threshold.
        Note:
        The powergrid environment is expected to be an instance variable of the
        class this method belongs to. The environment should have an output
        dictionary with keys "res_trafo.loading_percent" and "res_line.loading_percent".
        These represent the loading percentages of transformers and lines respectively.
        """
        overloaded_trafo_mask = self.kpi_raw["trafo_ts_data"] > self.transformer_max
        overloaded_lines_mask = self.kpi_raw["lines_ts_data"] > self.lines_max
        self.transformer_num_violations_dam = overloaded_trafo_mask.sum().sum()
        self.lines_num_violations_dam = overloaded_lines_mask.sum().sum()
        # Save data in a dictionary
        self.kpi_transformer_and_lines_violations = {
            "transformer_num_violations_dam": f"({self.transformer_num_violations_dam},"
            f"'{self.transformer_max}%')",
            "transformer_loading_avg": self.kpi_raw["trafo_node_statistics"].loc[
                "average"
            ][0],
            "transformer_loading_max": self.kpi_raw["trafo_node_statistics"].loc[
                "maximum"
            ][0],
            "transformer_loading_min": self.kpi_raw["trafo_node_statistics"].loc[
                "minimum"
            ][0],
            "transformer_loading_std": self.kpi_raw["trafo_node_statistics"].loc[
                "std_dev"
            ][0],
            "lines_num_violations_dam_sum": f"({self.lines_num_violations_dam.sum()},"
            f"'{self.lines_max}%'",
            "lines_loading_mean_avg": self.kpi_raw["lines_vm_ts_edges_statistics"]
            .loc["average"]
            .mean(),
            "lines_loading_mean_max": self.kpi_raw["lines_vm_ts_edges_statistics"]
            .loc["maximum"]
            .mean(),
            "lines_loading_mean_min": self.kpi_raw["lines_vm_ts_edges_statistics"]
            .loc["minimum"]
            .mean(),
            "lines_loading_mean_std": self.kpi_raw["lines_vm_ts_edges_statistics"]
            .loc["std_dev"]
            .mean(),
            "lines_loading_highest_max": self.kpi_raw["lines_vm_ts_edges_statistics"]
            .loc["maximum"]
            .max(),
            "lines_loading_lowest_min": self.kpi_raw["lines_vm_ts_edges_statistics"]
            .loc["minimum"]
            .min(),
        }

    def _calculate_power_losses_and_power_balance_kpi3(self):
        """
        Calculates KPI2: total grid losses, main and micro grid power for a
        given powergrid and simulation results.
        Returns:
        --------
        dict: A dictionary containing calculated KPIs.
        The dictionary keys are:
            'total_line_losses': Total line losses.
            'total_active_power_losses_transformer': Total active power losses in transformers.
            'total_res_load_p_mw': Total active load.
            'total_res_load_q_mvar': Total reactive load.
            'total_of_sgen_p_mw': Total generated active power.
            'total_of_sgen_q_mvar': Total generated reactive power.
            'total_of_storage_p_mw': Total power stored.
            'total_of_storage_p_mw': Total reactive power stored.
            'total_res_ext_grid_p_mw': Total active power supplied by the external grid.
            'total_res_ext_grid_q_mvar': Total reactive power supplied by the external grid.
            'total_grid_loss': Total grid losses (sum of line and transformer losses).
            'micro_grid_q': Difference between generated and consumed reactive power in micro grid.
            'main_grid_q': Total reactive power supplied by the main grid.
            'main_grid_p': Total active power supplied by the main grid.
            'micro_grid_p': Difference between generated and consumed active power in micro grid.
            'sum_grid': Total grid summary (negative sum of main and micro grid active power).
        Perform a short evaluation of the PowerGridEvaluator time-series analysis.
        This method reads and increments the sequence_id, saves the relevant data
        in a dictionary, creates a DataFrame from the dictionary, and saves the
        DataFrame as a CSV and Excel file with a formatted file name.
        """
        # Extract log variables from the power grid environment
        log_variables = self.powergrid_env.log_variables

        # Retrieve specific power grid characteristics from logged variables

        # Losses and supplies from lines and transformers
        line_loss_pl_mw = log_variables["res_line.pl_mw"]  # line_loss [MW]
        line_loss_ql_mvar = log_variables["res_line.ql_mvar"]  # line_loss [MVar]
        transformer_losses_p_mw = log_variables[
            "res_trafo.pl_mw"
        ]  # trafo_active power losses [MW]
        transformer_losses_q_mvar = log_variables[
            "res_trafo.ql_mvar"
        ]  # trafo_reactive power losses [MVar]
        # Power supplied from external grid and loads
        res_ext_grid_p_mw = log_variables[
            "res_ext_grid.p_mw"
        ]  # ext_grid_active_supply [MW]
        res_ext_grid_q_mvar = log_variables[
            "res_ext_grid.q_mvar"
        ]  # ext_grid_reactive_supply [MVar]
        res_load_p_mw = log_variables["res_load.p_mw"]  # load_active [MW]
        res_load_q_mvar = log_variables["res_load.q_mvar"]  # load_reactive [MVar]
        # Power generated by photovoltaic units (P and Q)
        sgen_p_mw = log_variables["sgen.p_mw"]  # pv_active [MW]
        sgen_q_mvar = log_variables["sgen.q_mvar"]  # pv_reactive [MVar]

        # Calculation of total power losses in MW and MVar
        total_line_losses_p_mw = line_loss_pl_mw.sum().sum()
        total_transformer_losses_p_mw = transformer_losses_p_mw.sum().sum()
        total_power_loss_p_mw = total_line_losses_p_mw + total_transformer_losses_p_mw
        total_line_losses_q_mvar = line_loss_ql_mvar.sum().sum()
        total_transformer_losses_q_mvar = transformer_losses_q_mvar.sum().sum()
        total_power_loss_q_mvar = (
            total_line_losses_q_mvar + total_transformer_losses_q_mvar
        )

        # Total power generated by PV units(P/Q)
        total_of_sgen_p_mw = sgen_p_mw.sum().sum()
        # Total reactive power loss in PV units
        total_of_sgen_q_mvar = sgen_q_mvar.sum().sum()

        # Check if storage scenario is not True
        if not self.powergrid_env.is_storage_scenario:
            # Set the values to zero
            total_of_storage_p_mw = 0
            total_of_storage_q_mvar = 0
            total_of_storage_discharge_power_loss = 0
        else:
            # Power supply from storage units
            storage_p_mw = log_variables["storage.p_mw"]  # storage_active
            storage_q_mvar = log_variables["storage.q_mvar"]  # storage_reactive
            total_of_storage_discharge_power_loss = (
                log_variables[
                    "storage.discharge_power_loss"
                ]  # storage_discharge_power_loss
                .sum()
                .sum()
            )
            # Total power stored by Storage units(P/Q)
            total_of_storage_p_mw = storage_p_mw.sum().sum()
            # Total reactive power loss in storage units
            total_of_storage_q_mvar = storage_q_mvar.sum().sum()
        # Total demand (P/Q)
        total_res_load_p_mw = res_load_p_mw.sum().sum()
        total_res_load_q_mvar = res_load_q_mvar.sum().sum()
        # Total power supplied by external grid (P/Q)
        total_res_ext_grid_p_mw = res_ext_grid_p_mw.sum().sum()
        total_res_ext_grid_q_mvar = res_ext_grid_q_mvar.sum().sum()
        # Calculation of total micro_grid power (P/Q)
        micro_grid_q_mvar = (
            total_of_sgen_q_mvar
            - total_res_load_q_mvar
            - total_of_storage_q_mvar
            - total_power_loss_q_mvar
        )
        micro_grid_p_mw = (
            total_of_sgen_p_mw
            - total_res_load_p_mw
            - total_of_storage_p_mw
            - total_power_loss_p_mw
        )
        # (P/Q) power from the main grid (external supply)
        main_grid_q_mvar = total_res_ext_grid_q_mvar
        main_grid_p_mw = total_res_ext_grid_p_mw
        # Calculate the power balance in the power grid.
        # This is the difference between the total power supply (main grid + microgrid)
        # and the power demand. If the power balance is not equal to zero, there is
        # either a surplus or deficit of power in the grid.
        # Negative power balance indicates a surplus of power (power supply > power demand).
        # Positive power balance indicates a power deficit (power supply < power demand).
        sum_grid_p_mw = -main_grid_p_mw - micro_grid_p_mw
        sum_grid_q_mvar = -main_grid_q_mvar - micro_grid_q_mvar
        num_of_sgen = int(len(self.powergrid_env.net.sgen))
        self.kpi_power_losses_and_power_balance = {
            "lines_loss_p_mw": total_line_losses_p_mw,
            "transformer_loss_p_mw": total_transformer_losses_p_mw,
            "grid_loss_p_mw": total_power_loss_p_mw,
            "total_line_losses_q_mvar": total_line_losses_q_mvar,
            "total_transformer_losses_q_mvar": total_transformer_losses_q_mvar,
            "total_power_loss_q_mvar": total_power_loss_q_mvar,
            "total_of_sgen_p_mw": total_of_sgen_p_mw,
            "total_of_sgen_q_mvar": total_of_sgen_q_mvar,
            "total_of_storage_p_mw": total_of_storage_p_mw,
            "total_of_storage_q_mvar": total_of_storage_q_mvar,
            "total_of_storage_discharge_power_loss": total_of_storage_discharge_power_loss,
            "total_res_load_p_mw": total_res_load_p_mw,
            "total_res_load_q_mvar": total_res_load_q_mvar,
            "total_res_ext_grid_p_mw": total_res_ext_grid_p_mw,
            "total_res_ext_grid_q_mvar": total_res_ext_grid_q_mvar,
            "micro_grid_q_mvar": micro_grid_q_mvar,
            "micro_grid_p_mw": micro_grid_p_mw,
            "main_grid_q_mvar": main_grid_q_mvar,
            "main_grid_p_mw": main_grid_p_mw,
            "sum_grid_p_mw": sum_grid_p_mw,
            "sum_grid_q_mvar": sum_grid_q_mvar,
            "num_of_sgen": f"{num_of_sgen:.1f}",
        }
        if self.powergrid_env.mpv_flag:
            mask_mpv = self.net.sgen.type == "MPV"
            num_of_mpv = mask_mpv.sum()
            mpvsgen_p_mw = log_variables["sgen.p_mw"].loc[
                :, mask_mpv
            ]  # MPV pv_active [MW]
            mpvsgen_q_mvar = log_variables["sgen.q_mvar"].loc[
                :, mask_mpv
            ]  # MPV pv_reactive [MVar]
            # Total power generated by MPV units(P/Q)
            total_of_mpvsgen_p_mw = mpvsgen_p_mw.sum().sum()
            # Total reactive power loss in MPV units
            total_of_mpvsgen_q_mvar = mpvsgen_q_mvar.sum().sum()
            mask_pv = self.net.sgen.type == "PV"
            num_of_pv = mask_pv.sum()
            pvsgen_p_mw = log_variables["sgen.p_mw"].loc[
                :, mask_pv
            ]  # PV pv_active [MW]
            pvsgen_q_mvar = log_variables["sgen.q_mvar"].loc[
                :, mask_pv
            ]  # PV pv_reactive [MVar]
            # Total power generated by only PV units(P/Q)
            total_of_pvsgen_p_mw = pvsgen_p_mw.sum().sum()
            # Total reactive power loss in only PV units
            total_of_pvsgen_q_mvar = pvsgen_q_mvar.sum().sum()
            num_of_load = int(len(self.powergrid_env.net.load))
            additional_kpis = {
                "num_of_mpv": f"{num_of_mpv:.1f}",
                "num_of_pv": f"{num_of_pv:.1f}",
                "num_of_load": f"{num_of_load:.1f}",
                "mpv_concentration_rate_percent": f"{self.powergrid_env.mpv_concentration_rate_percent:.2f}",
                "total_of_mpvsgen_p_mw": total_of_mpvsgen_p_mw,
                "total_of_mpvsgen_q_mvar": total_of_mpvsgen_q_mvar,
                "total_of_pvsgen_p_mw": total_of_pvsgen_p_mw,
                "total_of_pvsgen_q_mvar": total_of_pvsgen_q_mvar,
            }
            self.kpi_power_losses_and_power_balance.update(additional_kpis)

    def _calculate_storage_utilization_kpi4(self):
        """
        Calculates KPI2: total grid losses, main and micro grid power for a
        given powergrid and simulation results.
        """
        # Extract log variables from the power grid environment
        log_variables = self.powergrid_env.log_variables
        storages_self_discharge_pl_mw = log_variables[
            "storage.discharge_power_loss"
        ].sum()  # storage_discharge_power_loss
        storages_soc_ts_data = log_variables[
            "storage.soc_percent"
        ]  # storage_soc_percent
        storages_soc_ts_nodes_stats = self.calculate_ts_data_stats_kpi1(
            storages_soc_ts_data
        )
        # Hinzufügen der neuen Zeile "total_self_discharge_pl_mw" zu storages_soc_ts_nodes_stats
        storages_soc_ts_nodes_stats.loc["total_self_discharge_pl_mw"] = (
            storages_self_discharge_pl_mw
        )
        storages_soc_max_min_avg_nodes_stats = self.get_ts_statistics(
            data_frame=storages_soc_ts_data
        )
        # discharge_power_loss
        self.kpi_storage = {
            "storages_soc_nodes_stats": storages_soc_ts_nodes_stats,
            "storages_self_discharge_pl_mw": storages_self_discharge_pl_mw,
            "storages_soc_ts_max_min_avg_node_statistics": storages_soc_max_min_avg_nodes_stats,
        }

    def plot_vm_min_max_buses_kpi1(self, plot_name):
        """
        Creates a plot of the maximum and minimum bus voltage magnitudes with
        their corresponding bus numbers over time for a given simulation.
        The data for the plot is obtained from a dictionary called
        self.vm_stats. The function displays the plot in the current
        environment and does not return anything.
        add_plot_kpi1_vm_critic_limits:
            adds critic_limits additional to damage limits
        """
        # read data from dictionary buses_vm_ts_max_min_avg_bus_node_statistics
        data_frame = self.kpi_raw["buses_vm_ts_max_min_avg_bus_node_statistics"]
        # get mask to indices of all the lv buses in the grid
        lv_bus_mask = self.powergrid_env.lv_bus_mask
        # define bus_nums for limits for the second y-axis
        bus_nums = lv_bus_mask.index[lv_bus_mask.values].to_list()
        # Create a plot (ax1 for the first subplot and ax2 for the second subplot)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))

        color = ["k", "tab:red", "tab:blue", "tab:green", "tab:cyan"]

        ln1 = ax1.plot(
            data_frame.index,
            data_frame["maximum"],
            color=color[1],
            linestyle="-",
            label="Maximum Voltage Magnitude",
        )
        ln2 = ax1.plot(
            data_frame.index,
            data_frame["average"],
            color=color[0],
            linestyle="-",
            label="Average Voltage Magnitude",
        )
        ln3 = ax1.plot(
            data_frame.index,
            data_frame["minimum"],
            color=color[4],
            linestyle="-",
            label="Minimum Voltage Magnitude",
        )
        ln4 = ax2.plot(
            data_frame.index,
            data_frame["maximum_idx"],
            color=color[1],
            linestyle=":",
            label="Maximum Bus Number",
        )
        ln5 = ax2.plot(
            data_frame.index,
            data_frame["minimum_idx"],
            color=color[4],
            linestyle=":",
            label="Minimum Bus Number",
        )

        lns1 = ln1 + ln2 + ln3
        lns2 = ln4 + ln5
        labs1 = [l.get_label() for l in lns1]
        labs2 = [l.get_label() for l in lns2]
        ax1.legend(lns1, labs1, loc="best")
        ax2.legend(lns2, labs2, loc="best")

        ax1.set_xlabel("Timesteps", color=color[0])
        ax1.set_ylabel("Voltage Magnitude [pu]", color=color[0])
        ax2.set_ylabel("Bus Number", color=color[2])
        ax1.set_title("Voltage Magnitudes")
        ax2.set_title("Voltage Magnitudes Bus Numbers")

        ax1.tick_params(axis="y", labelcolor=color[0])
        ax2.tick_params(axis="y", labelcolor=color[2])
        ax1.set_xlabel("Time")

        ymin = bus_nums[0] - 2
        ymax = bus_nums[-1] + 1
        ax2.set_ylim([ymin, ymax])
        ax2.set_yticks(bus_nums)
        self.add_plot_kpi1_vm_critic_limits(ax1, color[3])

        fig.tight_layout()  # Adjust the layout to prevent overlap
        self.save_and_display_figure(fig_name=fig, plot_file_name=plot_name)

    def plot_vm_subplots_kpi1(self, plot_name):
        """Plots voltage magnitudes of buses in a grid of subplots.
        This function creates a 5x3 grid of subplots, each representing the voltage
        magnitude of a bus. The bus voltage magnitude data is retrieved from the
        environment's logged variables.
        Args:
            plot_name (str): The name of the plot to be saved.
        """
        # Retrieve bus voltage magnitudes [p.u] from logged variables
        bus_vm_log_variables = self.powergrid_env.log_variables["res_bus.vm_pu"]
        # Determine the shape of the data
        rows, columns = bus_vm_log_variables.shape
        # Check if the number of columns(buses) is greater than 15
        if columns > 15:
            # Calculate the average for each row if there are more than 15 columns
            row_averages = np.mean(bus_vm_log_variables, axis=1)
            # Plot the row averages
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(row_averages, marker="o")
            ax.set_title("Average Voltage Magnitudes per Bus")
            ax.set_xlabel("Bus Index")
            ax.set_ylabel("Average Voltage Magnitude (p.u.)")
        else:
            # Create a 5x3 grid of subplots for 15 or fewer columns
            fig, axs = plt.subplots(5, 3, figsize=(15, 20))
            # Set title for the entire figure
            fig.suptitle("Voltage Magnitudes of Buses")
            axs = axs.flatten()  # Flatten the 5x3 grid into a 15-element array
            for i, column in enumerate(bus_vm_log_variables.columns):
                axs[i].plot(bus_vm_log_variables[column])
                axs[i].set_title(f"BUS {column}")
                self.add_plot_kpi1_vm_critic_limits(axs[i], color="red")
            # Remove empty subplots
            if len(bus_vm_log_variables.columns) < len(axs):
                for i in range(len(bus_vm_log_variables.columns), len(axs)):
                    fig.delaxes(axs[i])
            # Improve layout
            fig.tight_layout()  # Adjust the layout to prevent overlap
        # Save and display the figure
        self.save_and_display_figure(fig_name=fig, plot_file_name=plot_name)

    def plot_lines_and_transformer_overloading(self, plot_name):
        """
        This function analyzes and plots the overloading of lines and transformers.
        It performs a statistical analysis including mean, minimum, maximum, and standard deviation.
        The number of instances exceeding the loading limits are also displayed.
        """
        # plot_name=self.plot_names[11]
        # Initialize the figure and axes
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(
            "Transformer and Lines Analysis"
        )  # Titel für die gesamte Figur setzen
        # Plot the transformer and lines data and statistical analysis
        self.plot_transformer_data(axs[0, 0], axs[0, 1])
        self.plot_lines_data(axs[1, 0], axs[1, 1])
        fig_manager = plt.get_current_fig_manager()
        fig.set_size_inches(18.5, 10.5, forward=False)
        self.save_and_display_figure(fig_name=fig, plot_file_name=plot_name)

    def add_plot_kpi1_vm_critic_limits(self, ax1, color="r"):
        """Adds to the figure the limits for the damaging and critical vm_pu."""
        ts_steps = round(self.powergrid_env.max_steps + 1)  # margin 1
        # Create a pandas dataframe from the max_damage_vm_pu list
        df_limit = pd.DataFrame(
            {
                "max_damage_vm_pu": [self.vm_scalar_limits["v_max_damage"]] * ts_steps,
                "min_damage_vm_pu": [self.vm_scalar_limits["v_min_damage"]] * ts_steps,
                "max_critic_vm_pu": [self.vm_scalar_limits["v_max_critic"]] * ts_steps,
                "min_critic_vm_pu": [self.vm_scalar_limits["v_min_critic"]] * ts_steps,
            }
        )
        ax1.plot(df_limit.max_damage_vm_pu, linestyle="--", color=color)
        ax1.plot(df_limit.min_damage_vm_pu, linestyle="--", color=color)

    def plot_transformer_data(self, ax1, ax2):
        """This function plots the transformer data and carries out a
        statistical analysis."""
        ax1.plot(self.kpi_raw["trafo_ts_data"], label="Transformer Data")
        ax1.set_title("Transformer Loading")
        ax1.set_ylabel("Transformer Loading (%)")

        stats_trafo = f"""
        max  : {self.kpi_raw['trafo_node_statistics'].loc["maximum"][0]:.2f}
        min  : {self.kpi_raw['trafo_node_statistics'].loc["minimum"][0]:.2f}
        avg : {self.kpi_raw['trafo_node_statistics'].loc["average"][0]:.2f}
        std  : {self.kpi_raw['trafo_node_statistics'].loc["std_dev"][0]:.2f}
        num_trafo_violations_dam: {self.transformer_num_violations_dam:.2f}
        """
        ax2.text(
            0.5,
            0.5,
            stats_trafo,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            bbox={"facecolor": "red", "alpha": 0.5},
        )
        ax2.axis("off")
        ax2.set_title("Analysis Transformer Loading")

    def plot_lines_data(self, ax1, ax2):
        """This function plots the line data and carries out a statistical
        analysis."""
        for column in self.kpi_raw["lines_ts_data"].columns:
            ax1.plot(self.kpi_raw["lines_ts_data"][column], label=column)

        ax1.set_title("Lines Loading")
        ax1.set_ylabel("Lines Loading (%)")

        # Create a DataFrame for the statistics
        stats_df = self.get_lines_stats_df()

        ax2.axis("tight")
        ax2.axis("off")
        table = ax2.table(
            cellText=stats_df.values,
            colLabels=stats_df.columns,
            rowLabels=stats_df.index,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax2.set_title("Analysis Lines Loading", fontsize=12, pad=20)

    def load_ctrl_mode_and_technical_params(self):
        """Sets selected standard mode parameters for the inverter."""
        # Define control and criticality(both voltage and grid) parameter keys.
        ctrl_mode_param_keys_1 = [
            "v_nom",
            "v_nom_net",
            "v_1",
            "v_2",
            "v_3",
            "v_4",
            "v_5",
        ]
        ctrl_mode_param_keys_2 = ["v_low_gain", "v_high_gain", "v_deadband_gain"]
        criticality_voltage_keys = [
            "v_min",
            "v_max",
            "v_min_max_delta",
            "v_crit_lower",
            "v_crit_upper",
            "v_delta_threshold",
            "v_max_threshold",
        ]
        criticality_grid_keys = ["transformer_max", "lines_max"]

        # Combine keys into single lists.
        control_param_keys = ctrl_mode_param_keys_1 + ctrl_mode_param_keys_2
        criticality_param_keys = criticality_voltage_keys + criticality_grid_keys

        # Assign attributes for control parameters, using .get() to avoid KeyError.
        for key in control_param_keys:
            setattr(self, key, self.selected_inverter_ctrl_params.get(key, None))
        # Same for criticality parameters.
        for key in criticality_param_keys:
            setattr(self, key, self.criticality_params.get(key, None))

    def compute_voltage_limits_kpi1(self):
        """Computes and returns voltage limits for damage and criticality as a
        DataFrame and a dictionary.
        The function loads the inverter's standard mode according to the parameters
        of the voltage regulation standard. It calculates the critical maximum and
        minimum values of the voltage as well as the maximum and minimum values for
        potential damage, storing them in a DataFrame.
        It also computes the damage and criticality thresholds for maximum and minimum
        voltage and stores these in a dictionary.
        Returns:
        df_vm_limits (pandas.DataFrame):
            A DataFrame that contains the maximum and minimum voltage values for
            criticality and damage.
        limits_vars (dict):
            A dictionary that contains the maximum and minimum damage and criticality
            thresholds for the voltage.
        """
        max_crit_vm_pu = (
            self.net.bus.min_vm_pu.values + 0.015 * 2 * self.net.bus.min_vm_pu.values
        )
        min_crit_vm_pu = (
            self.net.bus.max_vm_pu.values - 0.015 * 1 * self.net.bus.max_vm_pu.values
        )
        # Speichern der Variablen in einem DataFrame
        df_max_damage = pd.DataFrame(
            self.net.bus.max_vm_pu.values, columns=["max_damage_vm_pu"]
        )
        df_min_damage = pd.DataFrame(
            self.net.bus.min_vm_pu.values, columns=["min_damage_vm_pu"]
        )
        df_max_critic = pd.DataFrame(min_crit_vm_pu, columns=["max_critic_vm_pu"])
        df_min_critic = pd.DataFrame(max_crit_vm_pu, columns=["min_critic_vm_pu"])
        df_vm_limits = pd.concat(
            [df_max_damage, df_min_damage, df_max_critic, df_min_critic], axis=1
        )
        v_max_damage = self.v_nom + self.v_min_max_delta  # Maximum damage voltage limit
        v_min_damage = self.v_nom - self.v_min_max_delta  # Minimum damage voltage limit
        v_max_critic = self.v_crit_upper  # Maximum criticality voltage limit
        v_min_critic = self.v_crit_lower  # Minimum criticality voltage limit

        limits_vars = {
            "v_max_damage": v_max_damage,  # 'max_damage_vm_pu'
            "v_min_damage": v_min_damage,  # 'min_damage_vm_pu'
            "v_max_critic": v_max_critic,  # 'max_critic_vm_pu'
            "v_min_critic": v_min_critic,
        }  # 'min_critic_vm_pu'
        return df_vm_limits, limits_vars

    def assign_kpi1(self):
        """Calculate and assign KPI1 for the powergrid, including transformer,
        lines (edges), and bus voltages(buses) statistics and violations."""
        self.sim_arguments = {
            "sequence_id": self.sequence_id,
            "Simulation Start Date": self.powergrid_env.first_date_simulation,
            "Simulation End Date": self.powergrid_env.last_date_simulation,
            "Maximum Simulation Steps": f" {self.powergrid_env.max_steps}-steps",
            "Number of Simulation Days": f"{self.powergrid_env.max_steps//96}-days",
            "benchmark": self.powergrid_env.benchmark,
            "sb_code": self.powergrid_env.sb_code,
            "scenario": self.powergrid_env.scenario,
            "regulation_standard": self.powergrid_env.regulation_standard,
            "standard_mode": self.powergrid_env.standard_mode,
            "pv_ctrl": self.powergrid_env.args["control_modes"]["pv_ctrl"],
            "storage_p_ctrl": self.powergrid_env.args["control_modes"][
                "storage_p_ctrl"
            ],
            "storage_q_ctrl": self.powergrid_env.args["control_modes"][
                "storage_q_ctrl"
            ],
            "scaling_pv": self.powergrid_env.args["rawdata"]["scaling"]["pv"],
            "scaling_load": self.powergrid_env.args["rawdata"]["scaling"]["load"],
            "scaling_storage": self.powergrid_env.args["rawdata"]["scaling"]["storage"],
            "is_storage_scenario": self.powergrid_env.is_storage_scenario,
            "is_mpv_scenario": self.powergrid_env.mpv_flag,
        }
        # (1) Conduct a brief dictionary evaluation to summarize key analysis aspects.
        self.kpi_key = {
            "vm_lv_mean_avg": self.kpi_vm_violations["vm_lv_mean_avg"],
            "transformer_loading_avg": self.kpi_transformer_and_lines_violations[
                "transformer_loading_avg"
            ],
            "sequence_id": self.sim_arguments["sequence_id"],
            "sb_code": self.sim_arguments["sb_code"],
            "scenario": self.sim_arguments["scenario"],
            "is_storage_scenario": self.sim_arguments["is_storage_scenario"],
            "is_mpv_scenario": self.sim_arguments["is_mpv_scenario"],
            "regulation_standard": self.sim_arguments["regulation_standard"],
            "standard_mode": self.sim_arguments["standard_mode"],
            "pv_ctrl": self.sim_arguments["pv_ctrl"],
            "storage_p_ctrl": self.sim_arguments["storage_p_ctrl"],
            "storage_q_ctrl": self.sim_arguments["storage_q_ctrl"],
            "scaling_pv": self.sim_arguments["scaling_pv"],
            "scaling_load": self.sim_arguments["scaling_load"],
            "scaling_storage": self.sim_arguments["scaling_storage"],
            "grid_loss_p_mw": self.kpi_power_losses_and_power_balance["grid_loss_p_mw"],
            "total_of_sgen_p_mw": self.kpi_power_losses_and_power_balance[
                "total_of_sgen_p_mw"
            ],
            "total_of_sgen_q_mvar": self.kpi_power_losses_and_power_balance[
                "total_of_sgen_q_mvar"
            ],
            "total_res_ext_grid_p_mw": self.kpi_power_losses_and_power_balance[
                "total_res_ext_grid_p_mw"
            ],
        }

        # Update the key KPI dictionary with violations data
        self.kpi_key.update(self.kpi_vm_violations)
        self.kpi_key.update(self.kpi_transformer_and_lines_violations)

        # Prepare a dictionary for data presentation. It includes information
        # on KPIs, power loss and balance, simulation arguments, and bus voltage statistics
        dict_data = {
            "kpi_key": pd.DataFrame(self.kpi_key, index=[0]).transpose(),
            "kpi_vm_violations": pd.DataFrame(
                self.kpi_vm_violations, index=[0]
            ).transpose(),
            "kpi_trafo_and_lines_violations": pd.DataFrame(
                self.kpi_transformer_and_lines_violations, index=[0]
            ).transpose(),
            "kpi_power_loss_and_balance": pd.DataFrame(
                self.kpi_power_losses_and_power_balance, index=[0]
            ).transpose(),
            "sim_arguments": pd.DataFrame(self.sim_arguments, index=[0]).transpose(),
            "buses_vm_ts_nodes_statistics": pd.DataFrame(
                self.kpi_raw["buses_vm_ts_nodes_statistics"]
            ),
            "buses_vm_ts_node_statistics": pd.DataFrame(
                self.kpi_raw["buses_vm_ts_max_min_avg_bus_node_statistics"]
            ),
        }
        # Save in loop for all output file types
        print("\n-------------Write KPI Evaluation Results to Disk-------------\n")
        print("log_variables output_file_path:         \n")
        # If it's a storage scenario, add the storage statistics to the data dictionary
        if self.powergrid_env.is_storage_scenario:
            dict_data["kpi_storage_statistics"] = pd.DataFrame(
                self.kpi_storage["storages_soc_nodes_stats"]
            )
            print("with Storage Information         \n")
        self.handle_additional_output_file_types(
            file_name_base=self.file_name[0], dict_data=dict_data
        )

    @staticmethod
    def get_ts_statistics(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with the maximum and minimum values and their
        corresponding index over the time series (ts) simulation.
        Args:
        - df (pandas.DataFrame):
            DataFrame containing the values for each time step.
        Returns:
        - ts_statistics (pandas.DataFrame):
            DataFrame with the following columns:
            - "maximum": Maximum value over the time series
            - "maximum_idx": Index corresponding to the maximum value
            - "minimum": Minimum value over the time series
            - "minimum_idx": Index corresponding to the minimum value
            - "average": Average value over the time series
        """
        df_ts_max_values = pd.DataFrame({"maximum": data_frame.max(axis=1)})
        df_ts_max_index = pd.DataFrame({"maximum_idx": data_frame.idxmax(axis=1)})
        df_ts_min_values = pd.DataFrame({"minimum": data_frame.min(axis=1)})
        df_ts_min_index = pd.DataFrame({"minimum_idx": data_frame.idxmin(axis=1)})
        df_ts_avg_values = pd.DataFrame({"average": data_frame.mean(axis=1)})

        # Create DataFrame for time series statistics
        ts_statistics = pd.concat(
            [
                df_ts_max_values,
                df_ts_max_index,
                df_ts_min_values,
                df_ts_min_index,
                df_ts_avg_values,
            ],
            axis=1,
        )
        return ts_statistics

    @staticmethod
    def calculate_ts_data_stats_kpi1(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame containing statistical metrics such as average,
        maximum value, minimum value, and standard deviation over the time series(ts)
        simulation.
        Args:
        - df (pd.DataFrame):
            A pandas DataFrame containing measurements
        Returns:
        - ts_metrics (pd.DataFrame):
            A pandas DataFrame containing the following statistics:
            - "maximum": Maximum value over the time series for each element
            - "minimum": Minimum value over the time series for each element
            - "average": Average value over the time series for each element
            - "std_dev": standard deviation of values observed for each element
        """
        # Get the index of all columns
        cols_index = data_frame.columns.tolist()
        # Define the names of the different statistics to be calculated
        df_rows = ["maximum", "minimum", "average", "std_dev"]

        # Create a pandas DataFrame to store the statistics for each index
        ts_metrics = pd.DataFrame(columns=cols_index, index=df_rows)

        # Update the DataFrame with the calculated values
        ts_metrics.loc[df_rows[0]] = data_frame.max()
        ts_metrics.loc[df_rows[1]] = data_frame.min()
        ts_metrics.loc[df_rows[2]] = data_frame.mean()
        ts_metrics.loc[df_rows[3]] = data_frame.std()
        return ts_metrics

    @staticmethod
    def get_scalar_vr_statistics(buses_vm_pu_masked) -> dict:
        """
        Returns a pandas DataFrame with scalar statistics for voltage magnitude(vm)
        over a time series(ts) simulation :
            the (1) maximum and (2) minimum voltage magnitude values across all
            buses and time steps, the (3) mean voltage magnitude value of all
            buses/nodes over all time steps, and the(4) mean voltage magnitude (vm)
            value of each bus at each time step.
        Args:
        - buses_vm_pu_masked (pandas DataFrame):
            A DataFrame containing voltage magnitude values for each bus over
            multiple time steps, where missing data is masked.
        Returns:
        - bus_vm_scalar_statistics (dict):
            A dictionary containing the following scalar statistics:
            - max_vm_pu_buses (float):
                The maximum voltage magnitude value across all buses and time steps.
            - min_vm_pu_buses (float):
                The minimum voltage magnitude value across all buses and time steps.
            - mean_vm_total_time (float):
                The mean voltage magnitude value of all buses/nodes over all time steps.
            - mean_vm_pu_by_bus_and_time_step (pandas Series):
                The mean voltage magnitude value of each bus at each time step.
        """
        # Calculate scalar statistics for voltage magnitude over the time series simulation
        # max and min vm value across all buses and time steps
        max_vm_pu_buses = buses_vm_pu_masked.max().max()
        min_vm_pu_buses = buses_vm_pu_masked.min().min()

        # Get the indices of maxi and min voltage magnitudes values in the masked array
        max_row, max_col = np.where(buses_vm_pu_masked.values == max_vm_pu_buses)
        min_row, min_col = np.where(buses_vm_pu_masked.values == min_vm_pu_buses)
        # Mean voltage magnitude value of all buses/nodes over time steps
        avg_vm_total_time = buses_vm_pu_masked.mean().mean()

        # Save the scalar/DataFrame variables in a dictionary
        bus_vm_scalar_statistics = {
            "max_vm_pu_buses": max_vm_pu_buses,
            "max_vm_pu_time_step": max_row,
            "max_vm_pu_busno": max_col,
            "min_vm_pu_buses": min_vm_pu_buses,
            "min_vm_pu_time_step": min_row,
            "min_vm_pu_busno": min_col,
            "avg_vm_total_time": avg_vm_total_time,
        }
        return bus_vm_scalar_statistics

    def get_lines_stats_df(self):
        """This function generates a DataFrame with statistical data."""
        stats_df = pd.DataFrame(index=["min", "max", "avg", "std"])

        for column in self.kpi_raw["lines_ts_data"].columns:
            stats_df[column] = [
                round(self.kpi_raw["lines_ts_data"][column].min(), 2),
                round(self.kpi_raw["lines_ts_data"][column].max(), 2),
                round(self.kpi_raw["lines_ts_data"][column].mean(), 2),
                round(self.kpi_raw["lines_ts_data"][column].std(), 2),
            ]
        # Add an additional column for the minimum, maximum, average and standard deviation
        stats_df["critic"] = [
            round(stats_df.loc["min"].min(), 2),
            round(stats_df.loc["max"].max(), 2),
            round(stats_df.loc["avg"].mean(), 2),
            round(stats_df.loc["std"].std(), 2),
        ]
        # Add an additional column for the corresponding index names
        stats_df["idx"] = [
            stats_df.loc["min"].idxmin(),
            stats_df.loc["max"].idxmax(),
            None,  # There is no corresponding index for the average
            None,
        ]  # There is no corresponding index for the standard deviation
        return stats_df
