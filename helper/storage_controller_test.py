"""Storage Control test module."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colorbar
# Importing parent module
from .storage_controller import StorageController

class StorageControllerTest(StorageController):
    """
    A subclass of StorageController for testing purposes. 
    Manages test plots, checks and creates necessary directories, and informs
    about the test results location.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.output_data_path)
        # Check if the directory exists
        os.makedirs(self.output_data_path, exist_ok=True)
        self.update_meas = False
        self.add_vm_pu_noise = False
        attributes_storage = [
            'vm_pu_meas', 
            'mom_sn_mva','mom_p_mw','mom_bat_p_mw',
            'norm_mom_p_mw', 'norm_p_mw',
            'norm_min_q_mvar', 'norm_max_q_mvar','norm_q_mvar','raw_norm_q_mvar', 
            'sn_mva', 'p_mw', 'q_mvar','soc_percent'] #,'q_out'

        step_size = 0.001
        num_steps = 21
        active_power_range = np.linspace(self.min_p_mw, self.max_p_mw + 2*step_size, num_steps)
        num_steps = 21
        voltage_range = np.linspace(self.v_min, self.v_max, num_steps)
        # Create the output directory if it does not exist
        os.makedirs(self.output_data_path, exist_ok=True)
        # self.min_cos_phi = min_cos_phi
        self.load_power_factor_settings(test_mode=False)
        data_frames = []
        self.data_source = 0
        self.profile_name = 0

        # for v_meas in voltage_range: # Iterate over voltage range
        #     self.vm_pu_meas = v_meas
        iter_time=0
        self.net.storage.soc_percent[0] = 50.0
        self.soc_percent = self.net.storage.soc_percent[0]
        self.net.storage.soc_percent[0]
        
        for v_meas in voltage_range: # Iterate over voltage range
            self.vm_pu_meas = v_meas
            iter_time=0
            self.net.storage.soc_percent[0] = 50.0
            self.soc_percent = self.net.storage.soc_percent[0]
            # self.net.storage.soc_percent[0] = 50.0
            for mom_p_mw in active_power_range: # Iterate over power range
                # mom_p_mw=active_power_range[0]
                if iter_time ==43:
                    a=0
                if iter_time != 0:
                    # Update Storage SoC (state of charge) based on previous time step (last time step)
                    self.update_soc_from_previous(iter_time)
                    self.update_soc_from_self_discharge()
                self.last_time_step = iter_time
                self.mom_p_mw = mom_p_mw
                self.mom_bat_p_mw = self.mom_p_mw
                self.storage_control_mode = "rbc_load_battery_ctrl"
                # self.run_battery_p_control(net=self.net,
                #                            time=iter_time,
                #                            storage_control_mode=self.storage_control_mode)
                # Update and monitor storage power
                self.update_and_monitor_storage_power(desired_p_mw=self.mom_bat_p_mw,
                                                      time=iter_time)
                self.run_battery_q_control(net=self.net, 
                                           storage_q_control_mode=self.storage_q_control_mode)
                print("self.p_mw:", self.p_mw,"self.q_mvar:", self.q_mvar)
                self.write_to_net(self.net)
                print("self.p_mw:", self.p_mw,"self.q_mvar:", self.q_mvar)
                # self.time_step(net=self.net,
                #                time=iter_time)
                print(iter_time)
                print("attributes_storage",attributes_storage)
                # Create dataframe for each combination
                data_dict = {f'self.{attr}': [getattr(self, attr)] for attr in attributes_storage}
                # If additional_dict is provided, update data_dict with additional_dict
                # Create and return a DataFrame from data_dict
                data_frame = pd.DataFrame(data_dict, index=[0])
                # Add dataframe to the list
                data_frames.append(data_frame)
                iter_time = iter_time+1
        # Concatenate all dataframes and ignore index
        results_v = pd.concat(data_frames, ignore_index=True)

        print(f"Storage Test Results are in this path:\n"
              f"{self.output_data_path}")

    def run_rule_based_control_vde_tests(self, net):
        """ Runs the rule-based control VDE (Verband der Elektrotechnik) tests. 
        Description: This function performs the rule-based control VDE tests for
        various control modes and parameters. It iterates over voltage and power
        ranges, generates results, and draws P-Q and Q-V diagrams."""
        self.update_meas = False
        # Define common settings
        attributes_v = ['vm_pu_meas', 'norm_min_q_mvar', 'norm_max_q_mvar', 'mom_sn_mva',
                        'mom_p_mw', # 'mom_charge_discharge_p_mw',
                        'norm_mom_p_mw', 'norm_p_mw', 'norm_q_mvar',
                        'raw_norm_q_mvar', 'sn_mva', 'p_mw', 'q_mvar'] #,'q_out'
        attributes_p = ["mom_cos_phi", "norm_max_p_mw", "norm_mom_p_mw", "norm_sn_mva",
                        "norm_p_mw", "norm_q_mvar", "p_mw", "q_mvar", "mom_p_mw",
                        "mom_sn_mva", "sn_mva"]
        attributes_storage = ['vm_pu_meas', 'mom_p_mw',
                              # 'mom_charge_discharge_p_mw',
                              'p_mw', 'q_mvar', 'soc_percent'] #,'q_out'

        control_modes = {
            'power_factor_active_power_ctrl': {
                'names': ['df_pf_90', 'df_pf_95'],
                'min_cos_phi_values': [0.90, 0.95],
                'test_mode': [False, True]
            },
            'const_power_factor_active_power_ctrl': {
                'names': ['df_const_pf_100', 'df_const_pf_93', 'df_const_pf_90'],
                'min_cos_phi_values': [1.00, 0.93, 0.90],
                'test_mode': [False, False, False]
            }
        }
        step_size = 0.001
        num_steps = 200
        # min_cos_phi = 0.90
        active_power_range = np.linspace(self.min_p_mw, self.max_p_mw + 2*step_size, num_steps)
        voltage_range = np.linspace(self.v_min, self.v_max, num_steps)
        # Create the output directory if it does not exist
        os.makedirs(self.output_data_path, exist_ok=True)
        # self.min_cos_phi = min_cos_phi
        self.load_power_factor_settings(test_mode=False)
        data_frames = []
        for v_meas in voltage_range: # Iterate over voltage range
            self.vm_pu_meas = v_meas
            iter_time=0
            # self.net.storage.soc_percent[0] = 50.0
            for mom_p_mw in active_power_range: # Iterate over power range
                self.last_time_step = iter_time
                iter_time = iter_time+1
                self.mom_p_mw = mom_p_mw
                storage_control_mode = "rbc_load_battery_ctrl"
                self.run_battery_p_control(net=net,
                                           time=iter_time,
                                           storage_p_control_mode=self.storage_p_control_mode)
                self.run_battery_q_control(net=net, 
                                           storage_q_control_mode=self.storage_q_control_mode)
                print(iter_time)
                # Create dataframe for each combination
                data_dict = {f'self.{attr}': [getattr(self, attr)] for attr in attributes_storage}
                # If additional_dict is provided, update data_dict with additional_dict
                # Create and return a DataFrame from data_dict
                data_frame = pd.DataFrame(data_dict, index=[0])
                # Add dataframe to the list
                data_frames.append(data_frame)
        # Concatenate all dataframes and ignore index
        results_v = pd.concat(data_frames, ignore_index=True)

        # Iterate through each standard mode
        for mode in ['deadband', 'base']:
            self.pv_regulation_standard_mode = mode
            self.inverter_ctrl_params['standard_mode'] = mode
            # self.load_standard_mode(self.pv_regulation_standard_mode)
            self.selected_inverter_ctrl_params = self.inverter_ctrl_params['standard_mode']
            results_v = self._iterate_over_voltage_and_power(voltage_range,
                                                             active_power_range,
                                                             attributes_v,
                                                             net)
            # Draw the P-Q diagram
            self.draw_pqv_diagram(results_v)
            # Draw the P-Q diagram
            self.draw_qv_diagram(results_v)
            self.draw_pq_diagram(results_v)
            print("standard_mode:\n", self.inverter_ctrl_params[mode])
            print("self.pv_regulation_standard_mode:",
                  self.inverter_ctrl_params['standard_mode'],"\n")
        # Dictionary with control modes and corresponding DataFrame names and min_cos_phi values
        #----------------------------------------------------------------------
        dfs = {}
        for pv_control_mode, data in control_modes.items():
            self.pv_control_mode = pv_control_mode
            for name, min_cos_phi_value, test_mode in zip(data['names'],
                                                          data['min_cos_phi_values'],
                                                          data['test_mode']):
                self.min_cos_phi = min_cos_phi_value
                if test_mode:
                    self.load_power_factor_settings(test_mode=test_mode)
                    self.load_control_rule_settings(net=net,
                                                    element_type="storage")
                dfs[name] = self.sub_test_power_factor_control_mode(
                    active_power_range, attributes_p, pv_control_mode)
        excel_writer_output= self.output_data_path + '/output.xlsx'
        with pd.ExcelWriter(excel_writer_output, engine='xlsxwriter') as writer:
        # with pd.ExcelWriter(self.output_data_path + '/output.xlsx') as writer:
            for name, data_frame in dfs.items():
                data_frame.to_excel(writer, sheet_name=name)
        self.update_meas = True
        print("---TEST SUCCESFUL---")

    def sub_test_power_factor_control_mode(self, active_power_range, attributes_p, pv_control_mode):
        """
        Test the validity of the limits of operation for the power factor control modes.
        This includes control mode:
            (2) cos φ(P) - Power Factor/Active Power Characteristic Mode
                and pv_control_mode = 'power_factor_active_power_ctrl',
            as well as the
            (3) constant power factor control mode with fixed cos φ
                and pv_control_mode = 'const_power_factor_active_power_ctrl'.
        In both cases, the function tests for control and validity of the limits of operation.
        Args:
            active_power_range : Range of the active power.
            attributes_p: Attributes related to power.
            pv_control_mode: The control mode for photovoltaic power.
        Returns:
            pd.DataFrame: Results of the test.
        """
        results_p = self._iterate_over_power(active_power_range,
                                             attributes_p,
                                             pv_control_mode)
        # Draw the cos φ(P) diagram
        self.draw_cos_phi_p_diagram(results_p)
        # Draw the P-Q diagram
        self.draw_pq_diagram(results_p)
        return results_p

    def save_plot(self, plot_name, dpi=500):
        """Save the plot as a JPEG file."""
        filename = (
            "storage_test_"
            + self.regulation_standard + "_"
            + self.storage_p_control_mode + "_"
            + self.storage_q_control_mode
            + plot_name
            + str(self.min_cos_phi).replace(".", "")
            + ".jpeg"
        )
        plt.savefig(f"{self.output_data_path}/{filename}", dpi=dpi)
        # Display the plot
        plt.show()
        # Close all figure windows
        plt.close('all')

    def _iterate_over_power(self, active_power_range, attributes, pv_control_mode):
        """ Iterates over power, returns a combined dataframe. """
        results_list = []
        for mom_p_mw in active_power_range: # Iterate over power range
            self.mom_p_mw = mom_p_mw
            # -----------------------------------------------------------------
            self.constant_power_factor_active_power_ctrl(mom_p_mw=self.mom_bat_p_mw)
            # -----------------------------------------------------------------
            # Calculate additional power factor and add it to the dictionary
            additional_dict = {"p_ac": self.p_mw / self.min_cos_phi}
            # Create dataframe
            data_frame = self._create_dataframe(attributes, additional_dict)
            # Add dataframe to the list
            results_list.append(data_frame)
        # Concatenate all dataframes and ignore index
        return pd.concat(results_list, ignore_index=True)

    def _iterate_over_voltage_and_power(self, voltage_range, active_power_range, attributes, net):
        # attributes = attributes_v
        """ Iterates over voltage and power, returns a combined dataframe. """
        data_frames = []
        for v_meas in voltage_range: # Iterate over voltage range
            self.vm_pu_meas = v_meas
            for mom_p_mw in active_power_range: # Iterate over power range
                self.mom_p_mw = mom_p_mw
                self.voltage_reactive_power_control_mode(net, mom_p_mw)
                # Create dataframe for each combination
                data_frame = self._create_dataframe(attributes)
                # Add dataframe to the list
                data_frames.append(data_frame)
        # Concatenate all dataframes and ignore index
        # results_v = pd.concat(data_frames, ignore_index=True)
        return pd.concat(data_frames, ignore_index=True)

    def _create_dataframe(self, attributes, additional_dict=None):
        """ Creates a dataframe from class attributes and an additional dictionary."""
        # Create a dictionary with class attributes
        data_dict = {f'self.{attr}': [getattr(self, attr)] for attr in attributes}
        # If additional_dict is provided, update data_dict with additional_dict
        if additional_dict:
            data_dict.update(additional_dict)
        # Create and return a DataFrame from data_dict
        return pd.DataFrame(data_dict, index=[0])

    def _create_3d_draw(self, x_data, y_data, z_data, norm_base, cmap_base, vmin, vmax,
                        x_label, y_label, z_label, cbar_label):
        """Creates a 3D scatter plot."""
        # Create a 3D figure
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        # Plot data
        axes.scatter(x_data, y_data, z_data, c=norm_base, cmap=cmap_base, marker='o')
        # Create another scatter plot for colorbar showing actual values
        cax, _ = matplotlib.colorbar.make_axes(axes, location='right', pad=0.3)
        normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap_base,
                                                norm=normalize,
                                                orientation='vertical')
        # Set colorbar label
        cbar.set_label(cbar_label)
        # Set axis labels
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.set_zlabel(z_label)
        # Display the plot
        plt.show()

    def _gradient_3d_draw(self, norm, colors, ratios):
        """ Creates a gradient for 3D plot. """
        # Normalize the ratios
        norm_ratios = [norm(ratio) for ratio in ratios]
        # Create a list that defines the color gradient
        gradient = []
        for i, color in enumerate(colors):
            gradient.append((norm_ratios[i], color))
        return LinearSegmentedColormap.from_list('gradient', gradient, N=256)

    def draw_pqv_diagram(self, results):
        """
        This function generates a 3D plot based on the parameters voltage,
        active power and reactive power. It creates a range of values for active
        power and voltage, and calls the voltage_reactive_power_control_mode
        method in a nested loop to collect data and store it in a DataFrame.
        """
        # results = results_v
        x_data = results["self.vm_pu_meas"]
        y_data = results["self.norm_p_mw"]
        z_data = results["self.norm_q_mvar"]
        # Setting up the normalization functions and color mapping for P-Q-V
        # P
        norm_p = lambda p: (p - self.norm_min_p_mw) / (self.norm_max_p_mw - self.norm_min_p_mw)
        colors_p = ['red', 'orange', 'green', 'green', 'green', 'green']
        ratios_p = [self.norm_min_p_mw, 0.25, 0.5, 0.5, 0.75, self.norm_max_p_mw]
        cmap_p = self._gradient_3d_draw(norm_p, colors_p, ratios_p) #y,
        # Q
        norm_q = (lambda q: (q - self.norm_min_q_mvar) /
                            (self.norm_max_q_mvar - self.norm_min_q_mvar))
        colors_q = ['red', 'orange', 'green', 'green', 'orange', 'red']
        ratios_q = [self.norm_min_q_mvar, -0.15, 0, 0, 0.15, self.norm_max_q_mvar]
        cmap_q = self._gradient_3d_draw(norm_q, colors_q, ratios_q) # z,
        # V
        norm_v = lambda v: (v - self.v_min) / (self.v_max - self.v_min)
        colors_v = ['red', 'orange', 'green', 'green', 'green', 'orange','red']
        ratios_v = [self.v_min , self.v_1, self.v_2, self.v_3, self.v_4, self.v_5, self.v_max]
        cmap_v = self._gradient_3d_draw(norm_v, colors_v, ratios_v) # x,
        # Applying _create_3d_draw function to P-Q-V
        self._create_3d_draw(x_data, y_data, z_data, norm_p(y_data), cmap_p,
                             self.norm_min_p_mw, self.norm_max_p_mw,
                           'vm_pu', 'p_mw', 'q_mvar', cbar_label='p_mw')
        self.save_plot(plot_name="_pVoltWattVAr_Diagram_")
        plt.close() # Close the plot
        self._create_3d_draw(x_data, y_data, z_data, norm_q(z_data), cmap_q,
                             self.norm_min_q_mvar, self.norm_max_q_mvar,
                           'vm_pu', 'p_mw', 'q_mvar', cbar_label='q_mvar')
        self.save_plot(plot_name="_qVoltWattVAr_Diagram_")
        plt.close() # Close the plot
        self._create_3d_draw(x_data, y_data, z_data, norm_v(x_data), cmap_v,
                             self.v_min, self.v_max,
                             'vm_pu', 'p_mw', 'q_mvar', cbar_label='vm_pu')
        self.save_plot(plot_name="_vVoltWattVAr_Diagram_")
        plt.close() # Close the plot

    def draw_qv_diagram(self, results):
        """
        This function creates and plots the Q(V) characteristic curve diagram.
        It uses the matplotlib library for plotting, demonstrating the relationship
        between the voltage and the reactive power in pu and cos_phi values,
        respectively.
        Depending on the chosen 'self.pv_regulation_standard_mode',
        the modes can be:
            'self.pv_regulation_standard_mode', the modes can be:
                self.pv_regulation_standard:"vde"
                pv_regulation_standard_modes: 'deadband','base' or 'customised'
                self.pv_regulation_standard:"ieee"
                pv_regulation_standard_modes: 'a','b' or 'customised'
        Parameters:
        results (dict): Contains the simulation results.
        """
        # Create a maximized figure and plot the characteristic curve
        _, ax1 = plt.subplots(figsize=(18.5, 10.5))
        # Set title if pv_regulation_standard_mode is 'deadband'
        title = ('Q(V) Characteristic Curve - '
                 f'{self.regulation_standard.upper()} Standards '
                 f'with {self.regulation_standard.upper()} mode')
        plt.title(title)
        #----------------------------------------------------------------------

        # Plot q_control vs v_meas
        # results = results_v
        ax1.plot(results['self.vm_pu_meas'], results['self.norm_q_mvar'],
                 label='Q(V) with constraint')
        ax1.plot(results['self.vm_pu_meas'], results['self.raw_norm_q_mvar'],
                 label='Q(V)')
        ax1.legend()
        # Set labels for the plot
        ax1.set_xlabel('Voltage (pu)')
        ax1.set_ylabel("$Q^{DER}/S_{Emax}$")
        # Setting up axis limits and tick values
        tick_increment = self.norm_max_q_mvar / 5
        ax1.set_xlim(self.v_min, self.v_max)
        ax1.set_ylim(self.norm_min_q_mvar*self.c_axis_factor,
                     self.norm_max_q_mvar*self.c_axis_factor)
        ax1.set_xticks(np.arange(self.v_min, self.v_max, 0.01))
        yticks = np.arange(-tick_increment*5, tick_increment*6, tick_increment)
        yticklabels = [
            f'{tick:.3f}=$Q_{{max}} (under excited)$' if tick == self.norm_max_q_mvar
            else f'{tick:.3f}=$Q_{{max}} (over excited)$' if tick == self.norm_min_q_mvar
            else f'{tick:.3f}' for tick in yticks
        ]
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(yticklabels)
        #----------------------------------------------------------------------

        ax2 = ax1.twinx()
        # Plot q_control vs v_meas
        ax2.plot(results['self.vm_pu_meas'], results['self.norm_q_mvar'])
        ax2.set_ylabel(r"$\cos(\phi)$")
        ax2.set_yticks(yticks)
        # Zweite Achse für cos_phi
        yticklabels_cos = [round(np.cos(np.arctan(tick)),
                                 self.c_precision) for tick in yticks]
        ax2.set_yticklabels(yticklabels_cos)
        # Define point coordinates according to selected_mode
        if self.pv_regulation_standard_mode == 'deadband':
            # "low", "medium_low", # "medium", # "medium_high", # "high"
            x_data = [self.v_1, self.v_2, self.v_3, self.v_4, self.v_5]
            # Maximum Q, 3xNominal Q , Minimum Q
            y_data = [self.norm_max_q_mvar] + [self.nomi_q_mvar]*3 + [self.norm_min_q_mvar]
            # Add text labels for each point
            labels = [f'${key}$' for key in self.ctrl_mode_param_keys_1[2:]]
            # Create a scatter plot with the two points
            plt.scatter(x_data, y_data)
            for i, label in enumerate(labels):
                plt.text(x_data[i], y_data[i]+0.01,label)
            # Define points
            x_vertex_points = [self.v_2 + (self.v_2-self.v_1), self.v_4 - (self.v_5-self.v_4)]
            y_vertex_points = [self.norm_min_q_mvar, self.norm_max_q_mvar]
            vertex_labels = [
                f'$v_2$ $+{int(100*round((self.v_2-self.v_1), self.c_precision))}$ $v_n$%',
                f'$v_4$ $-{int(100*round((self.v_5-self.v_4), self.c_precision))}$ $v_n$%'
            ]
            for i, label in enumerate(vertex_labels):
                plt.text(x_vertex_points[i], y_vertex_points[i]+0.01, label)
            # Define points
            x_points = [self.v_1,
                        self.v_2 + (self.v_2-self.v_1),
                        self.v_5,
                        self.v_4 - (self.v_5-self.v_4)]
            y_points = [self.norm_max_q_mvar,
                        self.norm_min_q_mvar,
                        self.norm_min_q_mvar,
                        self.norm_max_q_mvar]
            # Plot points and fill area
            plt.fill(x_points, y_points, color='green', alpha=0.1)
            plt.plot(x_points, y_points, 'bo')
        plt.grid(True)
        plt.show()
        self.save_plot(plot_name="_Q(V)_diagram_")
        plt.close() # Close the plot

    def draw_pq_diagram(self, results):
        """
        Draw a PQ (active-reactive power) diagram.
        Parameters
        ----------
        results : pandas.DataFrame
            Dataframe containing active and reactive power values.
        net : pandapower.Network
            The pandapower network object.
        self.pid : int
            ID of the power generator.
        self.min_cos_phi : float
            Minimum cos(phi) value.
        self.max_q_mvar : float
            Maximum reactive power value in MVAR.
        self.sn_mva : float
            Apparent power value in MVA. ->net,self.rated_apparent_power
        self.reactive_power_performance_limit : float
            Reactive power performance limit.
        """
        # Create a figure and a single subplot
        fig, _ = plt.subplots()
        # Get the current figure manager and maximize the window
        fig_manager = plt.get_current_fig_manager()
        # Set the size of the figure
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5, forward=False)

        # Create the PQ diagram
        plt.plot(-results["self.norm_p_mw"],
                  results["self.norm_q_mvar"],
                  label="PQ points")
        # Draw a circle with the radius of the maximum apparent power
        circle1 = Circle((0, 0), 1, edgecolor="black", linestyle="--", fill=False,
                         label="Maximum apparent power")
        circle2 = Circle((0, 0), self.reactive_power_performance_limit, color="grey",
                         linestyle="--", fill=True,
                         label="Reactive power performance limit", alpha=0.2)
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        # update available_reactive_power
        self.available_reactive_power = self.calc_allowed_reactive_power(self.min_cos_phi)
        # Coordinates of the triangle vertices
        triangle_coordinates = [(-1 * self.min_cos_phi, -self.available_reactive_power),
                                (-1 * self.min_cos_phi, self.available_reactive_power),
                                (0, 0)]
        # Create and add a triangle
        triangle = Polygon(triangle_coordinates, edgecolor='red', facecolor='green', alpha=0.1)
        plt.gca().add_patch(triangle)
        # Add text to the vertices of the triangle
        point1_text = str(self.available_reactive_power)
        point2_text = str(-self.available_reactive_power)
        point3_text = str(self.reactive_power_performance_limit)
        plt.text(0, +self.available_reactive_power, point1_text, color="blue",
                 fontsize=12, verticalalignment="bottom", horizontalalignment="left")
        plt.text(0, -self.available_reactive_power, point2_text, color="blue",
                 fontsize=12, verticalalignment="bottom", horizontalalignment="left")
        plt.text(0, self.reactive_power_performance_limit, point3_text, color="grey",
                 fontsize=12, verticalalignment="bottom", horizontalalignment="left")
        plt.text(self.reactive_power_performance_limit, 0, point3_text, color="grey",
                 fontsize=12, verticalalignment="bottom", horizontalalignment="left")
        plt.plot([0, -1 * self.min_cos_phi],
                 [self.available_reactive_power, self.available_reactive_power],
                 color="blue", linestyle="--")
        plt.plot([0, -1 * self.min_cos_phi],
                 [-self.available_reactive_power, -self.available_reactive_power],
                 color="blue", linestyle="--")
        # Draw a vertical line at half of the maximum apparent power
        plt.axvline(x=-1*0.50, color="blue", linestyle="--")
        # Draw arrows for the x- and y-axes
        arrow_length = 1.06
        plt.arrow(0, 0, arrow_length, 0, head_width=0.05*arrow_length,
                  head_length=0.05*arrow_length, color="black")
        plt.arrow(0, 0, -arrow_length, 0, head_width=0.05*arrow_length,
                  head_length=0.05*arrow_length, color="black")
        plt.arrow(0, 0, 0, arrow_length, head_width=0.05*arrow_length,
                  head_length=0.05*arrow_length, color="black")
        plt.arrow(0, 0, 0, -arrow_length, head_width=0.05*arrow_length,
                  head_length=0.05*arrow_length, color="black")
        # Add text next to the arrows
        alignment_params = {
            "fontsize": 12,
            "verticalalignment": "center",
            "horizontalalignment": "center"
        }
        plt.text(arrow_length, arrow_length*0.15,
                 "Absorption\nP (positive)", **alignment_params)
        plt.text(-arrow_length, arrow_length*0.15,
                 "\nInjection\nP (negative)", **alignment_params)
        plt.text(0, +arrow_length*0.8,
                 "Absorption / Under Excited\n voltage-decreasing \nQ (positive)",
                 **alignment_params)
        plt.text(0, -arrow_length*0.8,
                 "Injection / Over Excited\n voltage-increasing\nQ (negative)",
                 **alignment_params)
        # Add labels and legend
        plt.xlabel("$P^{DER}/S_{Emax}$")
        plt.ylabel("$Q^{DER}/S_{Emax}$")
        plt.title("PQ-Diagram (Active-Reactive Power)")
        plt.grid(True) # Optional: show gridlines
        plt.legend()
        # Show the plot
        plt.show()
        # Save the plot as a JPEG file and display it
        self.save_plot(plot_name="_PQ_Diagram_")
        # Close the plot
        plt.close('all')

    def draw_cos_phi_p_diagram(self, results):
        """
        Draws the Standard Characteristic Curve (cos(gamma)-P/Pn) in a diagram.
        Parameters
        ----------
        results : pandas.DataFrame
            The dataframe containing the normalized active power values
            ("self.norm_mom_p_mw") and the cosine of the angle(powerfactor)
            between active power (P) and apparent power (S) ("self.mom_cos_phi").
        Returns
        -------
        None
            This function doesn't return anything; it shows the plot directly using plt.show().
            The plot includes the standard characteristic curve, labels for the points, gridlines,
            a vertical line at half the maximum apparent power, and custom y-axis ticks.
        """
        # Create a figure and a single subplot
        fig, axis = plt.subplots()
        # Get the current figure manager and maximize the window
        fig_manager = plt.get_current_fig_manager()
        # Set the size of the figure
        fig.set_size_inches(18.5, 10.5, forward=False)
        # Create the PQ diagram
        plt.plot(results["self.norm_mom_p_mw"], results["self.mom_cos_phi"], label=r"$\cos \gamma$")
        # Add text to the vertices of the triangle
        point1_text = str(self.min_cos_phi)
        point2_text = str(self.min_cos_phi)

        plt.text(0, self.min_cos_phi, point1_text, fontsize=12, verticalalignment="bottom",
                 horizontalalignment="left")
        plt.text(0, 1.1, point2_text, fontsize=12, verticalalignment="bottom",
                 horizontalalignment="left")
        plt.plot([0, 1], [self.min_cos_phi, self.min_cos_phi], color="black", linestyle="--")
        plt.plot([0, 1], [1.1, 1.1], color="black", linestyle="--")

        # Draw a vertical line at half of the maximum apparent power
        plt.axvline(x=1 * 0.50, color="blue", linestyle="--")

        # Add labels and legend
        plt.title(r"$\cos\gamma-P/P_n$ Standard Characteristic Curve ")
        plt.ylabel(r"$\cos \gamma$")
        plt.xlabel(r"$P/P_{Emax}$")
        plt.grid(True) # Optional: show gridlines
        plt.legend()
        # y-Achse
        axis.set_yticks([0.9, 1.0, 1.1])
        axis.set_yticklabels([0.9, 1.0, 0.9])
        # Show the plot
        plt.show()
        # Save the plot as a JPEG file and display it
        self.save_plot(plot_name="_PF_Curve_")
        # Close the plot
        plt.close('all')

# if __name__ == "__main__":
#     test_pv_controller = PVControllerTest()
