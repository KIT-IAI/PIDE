"""PV Control module."""
import os
import numpy as np
import pandapower as pp
from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import pandapowerNet
from typing import Optional, Union, Dict, Any

class PVController(Controller):
    """
    This class represents a PV Controller in a power network simulation.
    It simulates each time step in a time series simulation for the PV system,
    updates its active and reactive power values based on a given PV generation
    profile and corresponding selected algorithm.
    INPUT:
        **net** (attrdict) - Pandapower net
        **pid** (int) - ID of the PV unit that is controlled
        **pv_control_mode** (str, None) - Selected algorithm in config
        **regulation_standard** (str) - Regulation standard for the controller
        **data_source** (str) - Data source for the momentary active power
                                **p_mw** (float) 
                                (positive for generation)
        **profile_name** (str[]) - Power profile for the control in data_source
    OPTIONAL:
        **resolution** (int, 15) - Resolution of the control in 15 minutes.
        **scale_factor** (float, 1.0) - Scale factor for the control.
        **order** (int, 0) - The order of the controller in the overall control process.
        **level** (int, pid) - The priority level of the controller.
        **in_service** (bool) - Indicates if the element is currently active
        **recycle** (bool, True) - Re-use of internal-data in a time series loop.
        Settings can be made in the configuration file 
        Optional to be set in advance.
    Methods:
        time_step(net, time):
        Simulates a time step in a time-series simulation for the PV system,
        updating its active power values.
    Notes:
        gen, sgen, and ext_grid are modelled in the generator point of view. 
        Active power will therefore be postive for generation, and Reactive power 
        will be negative for underexcited behavior (Q absorption, decreases voltage)
        and positive for for overexcited behavior (Q injection, increases voltage).
    Key quantities:
        **cos_phi** (float, NaN) - power factor 
        **sn_mva** (float, None) - rated/nominal power of the generator
        **qmode** (str, optional): The mode of reactive power operation.
                   "underexcited" (Q absorption, decreases voltage) or
                   "overexcited" (Q injection, increases voltage).
                   Defaults to "underexcited".
        **pmode** (str, optional): The mode of active power operation.
                  "load" for load, or "gen" for generation. Defaults to "load".
    """
    def __init__(self,
             net: pandapowerNet, # replace with the actual type
             pid: int,
             pv_control_mode: str,
             regulation_standard: str,
             timeseries_ctrl: str,
             inverter_ctrl_params: Dict[str, Any],
             output_data_path: str,
             data_source: Optional[str] = None,
             profile_name: int = 0,
             resolution: int = 15,
             scale_factor: float = 1.0,
             order: int = 0,
             level: int = 0,
             test_folder_name: str = "pv_controller_tests",
             in_service: bool = True,
             recycle: bool = True,
             **kwargs: Any):
        # Call the parent's(Controller) init function
        super().__init__(net, in_service=in_service, recycle=recycle, order=order,
                         level=level, initial_powerflow = True,**kwargs)
        self.der_element_type = "sgen"
        self.timeseries_ctrl = timeseries_ctrl
        # Modularize the initialization into separate methods
        self.init_constants(scale_factor=scale_factor)
        self.read_pv_attributes(net=net, did=pid)
        self.init_power_parameters()
        self.init_voltage_measurements()
        self.init_der_control_mode_settings(net=net,
                                            der_control_mode=pv_control_mode,
                                            regulation_standard=regulation_standard,
                                            inverter_ctrl_params=inverter_ctrl_params)
        self.configure_output_path(output_data_path=output_data_path,
                                   folder_name=test_folder_name)
        # Load PF(power factor), Regulation and Control Settings
        self.load_power_factor_settings()
        self.load_control_rule_settings(net=net,
                                        element_type=self.der_element_type)
        self.load_ctrl_mode_and_technical_params()
        # Time Series Data Settings
        self.init_time_steps(resolution=resolution)
        self.init_data_source(net=net,
                              data_source=data_source,
                              profile_name=profile_name)

    def init_constants(self, scale_factor):
        """ Initializes the constants to be used in this class.
        Args:
            scale_factor (float): The factor by which to scale the input values.
        """
        # Define constants for the number of decimal places
        self.c_axis_factor = 1.1
        # Define constants for the number of decimal places
        self.c_precision = 4
        # Scaling factor for input values
        self.scale_factor = scale_factor

    def read_pv_attributes(self, net, did):
        """ Reads the PV attributes from the network.
        Args:
            net: The network object.
            did: The id of the DER unit.
            pid: The id of the PV unit.
        """
        # Assign the PV attributes from the network
        self.pid = did # id of controlled PV unit
        # Bus id where the PV system is connected
        self.bus = net.sgen.at[did, "bus"]
        # Bus name where the PV system is connected
        self.bus_name = net.bus.at[self.bus, "name"]
        # Identifying the type of PV unit 
        self.gen_type = net.sgen.at[did,"type"]
        # PV name
        self.name = net.sgen.at[did,"name"]
        # Active power of the PV system (positive for generation)
        self.p_mw = net.sgen.at[did,"p_mw"]
        # Reactive power of the PV system
        self.q_mvar = net.sgen.at[did,"q_mvar"]
        # Nominal power (rated_apparent_power) of the PV system
        self.sn_mva = net.sgen.at[did,"sn_mva"]
        self.rated_apparent_power = net.sgen.at[did,"sn_mva"]

        self.index = net.sgen.index[did]
        self.in_service = net.sgen.at[did,"in_service"]
        self.zone = net.sgen.at[did,"zone"]

    def init_power_parameters(self):
        """ Initializes the power parameters of the Bus unit. """
        # PV-Power Parameters VDE-AR-N 4105 or the IEEE Std 1547-2018 Settings
        # Power mode (generation)
        self.pmode = "gen"
        # Reactive power mode
        self.qmode = "underexcited"
        # Sign of acitve power
        self.p_sign = self.get_p_sign(self.pmode)
        # Sign of reactive power
        self.q_sign = self.get_q_sign(self.qmode)
        self.min_cos_phi = 0.90
        self.nomi_cos_phi = 1.0
        self.mom_cos_phi = 1.0
        self.calc_mom_cos_phi = 1.0
        # Initializing the limit for reactive power performance to 10%
        self.ten_prct_reactive_power_performance_limit = 0.10
        self.reactive_power_performance_limit = 0.10

    def init_voltage_measurements(self):
        """ Initializes the Bus id voltage measurements where the PV system
        is connected. """
        # Add Voltage Measurements
        self.add_vm_pu_noise = True
        self.update_meas = True

    def init_der_control_mode_settings(self, net, der_control_mode,
                                       regulation_standard, inverter_ctrl_params):
        """ Initializes the control mode of the PV unit.
        Args:
            net: The network object.
            pv_control_mode: The control mode of the PV unit.
            regulation_standard: The regulation standard for the PV system.
            inverter_ctrl_params: The control parameters for the inverter.
            output_data_path: The path for the output data.
        """
        # Running a power flow analysis
        pp.runpp(net)
        # Network object
        self.net = net
        # Base Inverter Params
        self.base_inverter_ctrl_params = inverter_ctrl_params
        # Control mode of the DER PV systems
        self.pv_control_mode = der_control_mode
        # Regulation standard for DER systems
        self.regulation_standard = regulation_standard
        # Inverter control mode parameters
        self.inverter_ctrl_params = inverter_ctrl_params[regulation_standard]
        # Regulation standard mode parameters
        self.regulation_standard_mode = inverter_ctrl_params[regulation_standard]['standard_mode']
        # Selected Inverter Control and Criticality parameters
        self.selected_inverter_ctrl_params = self.inverter_ctrl_params[
            self.regulation_standard_mode]
        self.criticality_params = inverter_ctrl_params["criticality"]
        self.applied = False

    def configure_output_path(self, output_data_path, folder_name):
        """ Configures the output paths.
        Args:
            output_data_path: The base path for the output data.
        """
        # Prepare output path
        self.output_data_path = output_data_path
        self.output_elm_control_path = os.path.join(output_data_path, folder_name)

    def init_time_steps(self, resolution):
        """ Initializes the time steps.
        Args:
            resolution: The resolution of the time steps.
        """
        # TimeSteps settings(default: 15min)
        self.resolution = resolution
        # Time constants "c" settings
        self.c_hours_in_mins = 60 # default: 15min values
        self.c_days_in_hours = 24 # default: 15min values
        self.steps_per_hour = self.resolution / self.c_hours_in_mins

    def init_data_source(self, net, data_source, profile_name):
        """ Initializes the data source for the PV unit.
        Args:
            net: The network object.
            data_source: The data source for the time series values.
            profile_name: The name of the profile in the data source.
        """
        # Data source for time series values
        self.data_source = data_source
        self.profile_name = profile_name
        # Scaling for the PV unit
        self.scaling = net.sgen.at[self.pid,"scaling"]
        # Initialize the last time step as None
        self.last_time_step = None

    def load_power_factor_settings(self, test_mode=False):
        """
        Set cos_phi values (underexcited & overexcited) based on the rated apparent
        power of DER, following the regulation standards for DERs (PVs, ESS, EVCS).
        The regulations can be either German VDE Technical Low Voltage Regulations
        (VDE-AR-N 4105) or IEEE Std 1547-2018.
        This function checks if the rated apparent power is <= 0.0046 MVA (4.6 kVA),
        and sets the underexcited & overexcited cos_phi values accordingly.
        DER-Unit sn_mva <= 0.0046 MVA (4.6 kVA):
            For DERs with a rated apparent power of sn_mva <= 0.0046 MVA, 
            the grid operator provides a fixed cos_phi value between 
            cos_phi = 0.95 lagging and cos_phi = 0.95 leading.
        DER-Unit sn_mva > 0.0046 MVA (4.6 kVA):
            For DERs with a rated apparent power of sn_mva > 0.0046 MVA,
            the grid operator specifies either set to self.v_min.
        Args:
            test_mode (bool):
                Flag to bypass the rated apparent power check for testing purposes.
                Default is False. If set to True, the check for 
                self.sn_mva <= 0.0046 is skipped.
        Attributes:
            self.regulation_standard (str):
                Standard for cos_phi settings.
                Can be either "VDE-AR-N 4105" or "IEEE Std 1547-2018".
                https://www.vde-verlag.de/standards/0100492
                doi: 10.1109/IEEESTD.2018.8332112
                This attribute determines the rules for setting cos_phi values.
        Where:
            - Pmom (or P) is the momentary active power
            - PEmax is the maximum active power
            - Q is the reactive power
        """
        # Check DERs for setting cos_phi values
        if self.regulation_standard.lower() == "vde":
            if test_mode:
                # Settings for test mode
                self.min_cos_phi = 0.900
                self.available_reactive_power = self.calc_allowed_reactive_power(self.min_cos_phi)
                self.reactive_power_performance_limit = 0.200
            elif self.sn_mva <= 0.001:
                # Settings for DERs <= 1000 VA 
                self.min_cos_phi = 1.00 # due that the MPV reactive power is set to zero
                self.available_reactive_power = self.calc_allowed_reactive_power(self.min_cos_phi)
                # pure active power
                self.reactive_power_performance_limit = 0.00
            elif self.sn_mva <= 0.0046 and self.sn_mva > 0.001:
                # Settings for 1000 VA <= DERs <= 4600 VA
                self.min_cos_phi = 0.950
                self.available_reactive_power = self.calc_allowed_reactive_power(self.min_cos_phi)
                # If 0 ≤ P / PEmax < 0.2, then DER terminals Qmax ≤ 0.2 * PEmax
                self.reactive_power_performance_limit = 0.200
            else:
                # Settings for DERs > 4600 VA
                self.min_cos_phi = 0.900
                self.available_reactive_power = self.calc_allowed_reactive_power(self.min_cos_phi)
                # If 0 ≤ P / PEmax < 0.1, then DER terminals Qmax ≤ 0.1 * PEmax
                self.reactive_power_performance_limit = 0.100
        elif self.regulation_standard.lower() == "ieee":
            # Add implementation for IEEE Std 1547-2018 here
            pass
        else:
            raise ValueError("Unsupported standard. Supported standards are"
                             " 'VDE-AR-N 4105' and 'IEEE Std 1547-2018'.")

    def load_ctrl_mode_and_technical_params(self):
        """ Sets selected standard mode parameters for the inverter."""
        # Define control and criticality(both voltage and grid) parameter keys.
        self.ctrl_mode_param_keys_1 = ["v_nom","v_nom_net","v_1","v_2","v_3","v_4","v_5"]
        ctrl_mode_param_keys_2 = ["v_low_gain","v_high_gain","v_deadband_gain"]
        criticality_voltage_keys = ["v_min", "v_max", "v_min_max_delta",
                                    "v_crit_lower","v_crit_upper",
                                    "v_delta_threshold","v_max_threshold"]
        criticality_grid_keys = ["transformer_max", "lines_max"]

        # Combine keys into single lists.
        control_param_keys = self.ctrl_mode_param_keys_1 + ctrl_mode_param_keys_2
        criticality_param_keys = criticality_voltage_keys + criticality_grid_keys

        # Assign attributes for control parameters, using .get() to avoid KeyError.
        for key in control_param_keys:
            setattr(self, key, self.selected_inverter_ctrl_params.get(key, None))
        # Same for criticality parameters.
        for key in criticality_param_keys:
            setattr(self, key, self.criticality_params.get(key, None))

    def load_control_rule_settings(self, net, element_type):
        """
        Load control settings for the given element based on network attributes
        and guidelines. Sets the necessary element attributes for calculations, such
        as max/min active and reactive power injections, and voltage magnitude 
        limits. Sets voltage deviation limits per unit.
        Args:
            net (attrdict): Pandapower network
            element_type (str): The type of the element (e.g., 'sgen' for PV systems,
                                'storage' for storage systems)
            element_id (int): The id of the element in the network
        Notes: Read controllable element attributes from net - necessary for OPF
        
        Raises:
            ValueError: If an unsupported element type is provided.
        """
        if element_type not in ['sgen', 'storage']:
            raise ValueError(f"Unsupported element type: {element_type}."
                             "Supported types: 'sgen', 'storage'.")
        # Load general distributed energy resources (DER) settings
        self._load_der_settings()
        self._load_sgen_settings(net)
        # Load settings for voltage magnitude
        self._load_voltage_magnitude_settings(net)

    def _load_der_settings(self):
        """ Load general Distributed Energy Resource (DER) settings. """
        # Minimum active power
        self.min_p_mw = 0 # Set to zero since it's a PV system
        # Maximum active power
        self.max_p_mw = self.sn_mva
        # Momentary acitve, reactive and apparent power
        self.mom_p_mw = 0 # raw since unprocessed or measured size
        self.mom_q_mvar = 0
        self.mom_sn_mva = 0
        # Norm Maximum/Minimum reactive power (Q) injection/absorption
        self.norm_min_q_mvar = -self.available_reactive_power
        self.norm_max_q_mvar = +self.available_reactive_power
        self.min_q_mvar = self.norm_min_q_mvar * self.sn_mva
        self.max_q_mvar = self.norm_max_q_mvar * self.sn_mva
        self.nomi_q_mvar = 0
        self.norm_max_p_mw = self.max_p_mw/self.max_p_mw
        self.norm_min_p_mw = 0
        self.norm_sn_mva = self.sn_mva/self.sn_mva
        self.norm_p_mw = None
        self.norm_q_mvar = None
        self.norm_mom_p_mw = None
        self.norm_mom_q_mvar = None

        # Save the unbounded reactive power for voltage control
        self.raw_norm_q_mvar = 0
        # Define active power thresholds for cosinus phi control per VDE/IEEE rules.
        self.norm_p_mw_lower_threshold = 0.5 #50.0
        self.norm_p_mw_upper_threshold = 1.0 #100.0

        # Initial voltage deviation limits in per-unit (PU)
        self.lower_vm_pu = 0.95
        self.upper_vm_pu = 1.05

        # Initial PV control parameters
        self.v_1 = None
        self.v_2 = None
        self.v_3 = None
        self.v_4 = None
        self.v_5 = None
        self.v_min = None
        self.v_max = None
        self.vm_pu_meas = None

    def _load_voltage_magnitude_settings(self, net):
        """ Load voltage magnitude settings from the network, defaulting to 
        predefined limits.
        Args:
            net (pandapowerNet): Pandapower network data structure.
        """
        # Maximum and Minimum voltage magnitude per-unit (PU)
        # Use value from net if available, else use upper voltage deviation
        self.max_vm_pu = (
            net.bus.at[self.bus, "max_vm_pu"]
            if not np.isnan(net.bus.at[self.bus, "max_vm_pu"])
            else self.upper_vm_pu)
        # Use value from net if available, else use lower voltage deviation
        self.min_vm_pu = (
            net.bus.at[self.bus, "min_vm_pu"]
            if not np.isnan(net.bus.at[self.bus, "min_vm_pu"])
            else self.lower_vm_pu)

    def _load_sgen_settings(self, net):
        """ Load control settings specific for the 'sgen' elements. """
        # PV specific code
        # Minimum/Maximum reactive power (Q) injection/absorption
        # Minimum (discharging) reactive power
        self.min_q_mvar = (net.sgen.at[self.pid, "min_q_mvar"]
                           if "min_q_mvar" in net.sgen.keys() else self.min_q_mvar)
        # Maximum (charging) reactive power
        self.max_q_mvar = (net.sgen.at[self.pid, "max_q_mvar"]
                           if "max_q_mvar" in net.sgen.keys() else self.max_q_mvar)
        # Overwrite inverter (pid) values following VDE guidelines
        net.sgen.at[self.pid, "max_q_mvar"] = self.max_q_mvar
        net.sgen.at[self.pid, "min_q_mvar"] = self.min_q_mvar
        net.sgen.at[self.pid, "max_p_mw"] = self.max_p_mw
        net.sgen.at[self.pid, "min_p_mw"] = self.min_p_mw

    def calc_allowed_reactive_power(self, cos_phi_value):
        """Calculate the available reactive power ("zur Verfügung stehende
        Blindleistung") (q_available) for a given power factor (cos φ).
        This function calculates the phase angle (φ) and the tangent of
        the phase angle (tan φ) based on the given power factor (cos φ).
        It then multiplies the tangent of the phase angle by the power 
        factor to obtain the Qvb value, which represents the available 
        reactive power in an electrical system, typically related to 
        the apparent power (S).
        Args:
            cos_phi_value (float): The power factor (cos φ) value.
        Returns:
            float: The calculated q_available value.
        Notes: 
            q_available_095 = self.calc_allowed_reactive_power(0.95)
            print("Qvb for cos φ = 0.95:", q_available_095)
        """
        # Calculate the phase angle φ
        phi = np.arccos(cos_phi_value)
        # Calculate the tangent of the phase angle
        tan_phi = np.tan(phi)
        # Multiply the tangent of the phase angle by the power factor
        q_available = tan_phi * cos_phi_value
        return round(q_available, self.c_precision)


    def get_p_sign(self, pmode):
        """ Calculates the P-Sign value based on the given P mode.
        Args:
            pmode (str): The P mode ("load" or "gen").
        Returns:
            psign (bool): The P-Sign value."""
        if pmode == "load":
            psign = 1
        elif pmode == "gen":
            psign = -1
        else:
            raise ValueError(f'Unknown mode {pmode} - specify "load" or "gen"')
        return psign

    def get_q_sign(self, qmode):
        """ Calculates the Q-Sign value based on the given Q mode.
        Args:
            qmode (str): The Q mode "ind"("underexcited") or "cap"("overexcited").
        Returns:
            qsign (bool): The Q-Sign value."""
        if qmode in ("ind", "underexcited"):
            qsign = 1
        elif qmode in ("cap", "overexcited"):
            qsign = -1
        else:
            raise ValueError(
                f'Unknown mode {qmode} - specify "underexcited" (Q absorption, decreases voltage)'
                f' or "overexcited" (Q injection, increases voltage)')
        return qsign

    def add_voltage_noise(self, net, voltage_error_std_dev=0.001):
        """ Adds Gaussian noise to the voltage measurement of a bus in a power grid.
        Args:
            net (pandapowerNet): The pandapower network object.
            voltage_error_std_dev (float): standard deviation of voltage error. 
                                           Default: 0.001-0.01, i.e., 0.1%-1%.
        Attributes Updated:
            vm_pu (float): Actual voltage magnitude at the bus.
            vm_pu_meas (float): Noisy voltage magnitude if add_vm_pu_noise is True; 
                                otherwise, it is set to the actual voltage magnitude.
            loc (float): Mean of voltage error. Default: 0.001-0.01 (0.1%-1%).
            error_std_dev (float): Standard deviation of voltage error. Default: 0.001-0.01 (0.1%-1%).
        If add_vm_pu_noise is True, Gaussian noise with mean loc and 
        standard deviation error_std_dev is added to the voltage magnitude.
        """
        # Load flow calculation pp.runpp(net)
        self.vm_pu = net.res_bus.at[self.bus, "vm_pu"] # self.v_pcc
        if self.add_vm_pu_noise:
            # Generate Gaussian noise with loc_X=self.loc and X_std=self.error_std_dev:
            self.voltage_noise = np.random.normal(loc=0, scale=voltage_error_std_dev)
            # Apply error to the voltage measurement
            self.vm_pu_meas = self.vm_pu + self.voltage_noise
        else:
            self.vm_pu_meas = self.vm_pu

    def _check_reactive_power_tolerance(self, tolerance=0.02):
        """ Check if the calculated reactive power is within the specified 
        tolerance range of max_q_mvar. If the calculated reactive power is 
        outside the range, a message will be printed indicating that it is
        outside the tolerance.
        Args:
            tolerance (float, optional): The allowed percentage difference.
            Defaults to 0.02 (2%). """
        # Calculate reactive power
        calc_q_mvar = round((self.norm_sn_mva**2 - self.norm_mom_p_mw**2)**0.5, self.c_precision)
        # Calculate the lower & upper limit for acceptable reactive power
        lower_limit = self.norm_max_q_mvar * (1 - tolerance)
        upper_limit = self.norm_max_q_mvar * (1 + tolerance)
        if not lower_limit <= calc_q_mvar <= upper_limit:
            print("calc_q_mvar is outside the 2% tolerance of max_q_mvar.")

    def is_converged(self, net):
        """
        Checks if the controller has already been applied and the power flow
        solution has converged.
        Args:
            net (pandapowerNet): The pandapower network object.
        Returns:
            converged (bool): True if the controller has been applied and the power
                              flow solution has converged, False otherwise.
        Notes:
            This function checks if the 'applied' attribute of the controller
            object is True, indicating that the controller has already been applied.
            It also checks if the power flow solution in the network has converged
            using the 'ppc' attribute of the pandapower network object. If the
            power flow solution has not converged, this function will return
            False even if the controller has been applied.
        """
        return self.applied and net["_ppc"]["success"]

    def write_to_net(self, net):
        """
        Writes the active and reactive power output setpoint of the PV unit to
        the pandapower network object.
        Args:
            net (pandapowerNet): The pandapower network object.
        Notes:
            This function updates the 'p_mw' and 'q_mvar'
            attributes of the PV unit with the controller's current values.
            The 'pid' attribute of the controller object is used to locate the
            corresponding PV unit in the network. The changes are made
            directly to the 'sgen' dataframe within the network object.
        """
        # if self.gen_type =="MPV":
        #     print(f"self.gen_type: {self.gen_type}")
        #     print(f"self.p_mw: {self.p_mw}")
        #     print(f"self.q_mvar: {self.q_mvar}")
        # Write updated active(P), reactive(Q) power setpoint to the pandapower net
        net.sgen.at[self.pid,"p_mw"] = self.p_mw
        # Write updated voltage setpoint to the pandapower net
        net.sgen.at[self.pid,"q_mvar"] = self.q_mvar

    def control_step(self, net):
        """
        Executes a control step by writing the current control values to the
        network object.
        Args:
            net (pandapowerNet): The pandapower network object.
        Notes:
            This function updates the active and reactive power output of the PV 
            unit in the network object by calling the 'write_to_net' function. 
            It also sets the 'applied' attribute of the controller object to True,
            indicating that the values set in the previous control step have been
            included in the load flow calculation. This is useful for determining
            whether the controller has already been applied to the network in 
            subsequent control steps.
        """
        self.write_to_net(net)
        self.applied = True

    def _reset_applied(self):
        """Resets the applied variable to False."""
        self.applied = False

    def time_step(self, net, time):
        """ Executes a time step by updating the controller state and writing 
        the current control values to the network object.
        Args:
            net (pandapowerNet): The pandapower network object.
            time (int or float): The current time step.
        Notes:
            This function first resets the 'applied' attribute of the controller
            object to False. It then updates the 'last_time_step' attribute to the
            current time step. If a data source and profile name are provided, the
            function runs the PV control based on the specified PV control mode.
            Finally, it writes the active and reactive power output of the PV unit
            to the pandapowerNet object by calling the 'write_to_net' function.
        """
        self._reset_applied()
        self.last_time_step = time
        # if time == 45:
        #     print("a")
        # read new values from a profile
        # Update step (last time step)
        if self.data_source:
            if self.profile_name is not None:
                # Get active power values for current time step
                self.read_active_power_values(time)
                # Run pv control based on the specified pv control mode
                self.run_pv_control(net, time, self.pv_control_mode)
            else:
                pass
        else:
            raise ValueError("Invalid PV Control Mode specified")
        # Writes p_mw and q_mvar of the PV unit to the net
        self.write_to_net(net)

    def run_pv_control(self, net, time, pv_control_mode):
        """
        Local Voltage and Reactive/Active Power Control Strategies in Power 
        Distribution Networks requirements for inverter based I-DERs normal 
        operating performance.
        """
        if pv_control_mode == "datasource" or self.gen_type == "MPV":
            # (3) Raw-Databased Control(RDC) of PV Energy Unit:
            self.raw_databased_control()
        elif self.regulation_standard.lower() == "vde":
            # (1) Rule-Based-Control (RDC) according to VDE guidelines:
            self.rule_based_control_vde(net, pv_control_mode)
        elif self.regulation_standard.lower() == "ieee":
            # (2) Raw-databased Control(RDC) according to IEEE guidelines:
            self.rule_based_control_ieee(pv_control_mode)
        else:
            raise ValueError("Oops! That was no valid pv control and regulation standard mode")
        # Enforce the limits on active and reactive power for the PV system
        self.limit_pv_power_output()
        # Monitor PV control if the power level is above 50% of the specified value
        # if self.norm_mom_p_mw > self.norm_p_mw_lower_threshold:
        #     self.monitor_pv_control(time)
        if self.q_mvar > 0:
            self.monitor_pv_control(time)

    def read_active_power_values(self, time):
        """ Read active power values from a raw data profile of the sgen element
        and set reactive power to zero.
        Args:
            time (int): The current time step.
        """
        self.mom_p_mw = self.data_source.get_time_step_value(time_step=time,
                                                             profile_name=self.profile_name,
                                                             scale_factor=self.scale_factor)
        self.mom_q_mvar = 0
        self.norm_mom_p_mw = self.mom_p_mw / self.sn_mva
        self.norm_mom_q_mvar = 0

    def raw_databased_control(self):
        """ Sets the active power (p_mw) and reactive power (q_mvar) based on the
        instantaneous power values obtained from the raw data control strategy.
        """
        self.p_mw = self.mom_p_mw
        self.q_mvar = self.mom_q_mvar

    def rule_based_control_vde(self, net, pv_control_mode):
        """ Performs rule-based control based on the specified VDE PV control mode.
        Args:
            net (pandapowerNet): The pandapower network object.
            pv_control_mode: The PV control mode to use.
        """
        if pv_control_mode == "voltage_reactive_power_ctrl":
            # (1):'voltage_reactive_power_ctrl'
            self.voltage_reactive_power_control_mode(net)
        elif pv_control_mode == "power_factor_active_power_ctrl":
            # (2): 'power_factor_active_power_ctrl'
            self.power_factor_active_power_ctrl()
        elif pv_control_mode == "constant_power_factor_active_power_ctrl":
            # (3): 'constant_power_factor_active_power_ctrl'
            self.constant_power_factor_active_power_ctrl()
        elif pv_control_mode == "yyy_ctrl":
            # (4): Add new implementation ->...
            pass
        else:
            raise ValueError("Oops! That was no valid pv control and regulation standard mode")
        return self.p_mw, self.q_mvar

    def rule_based_control_ieee(self, pv_control_mode):
    # def rule_based_control_ieee(self, net, time, pv_control_mode):
        """ Executes rule-based control according to the specified IEEE PV control mode.
        Args:
            net (pandapowerNet): The pandapower network object.
            pv_control_mode: The PV control mode to use.
        """
        # IEEE (2):
        if pv_control_mode == "constant_power_factor_active_power_ctrl":
            # (1):'constant_power_factor_active_power_ctrl'
            self.constant_power_factor_active_power_ctrl()
        elif pv_control_mode == "voltage_reactive_power_ctrl":
            # (2): 'voltage_reactive_power_ctrl'
            # self.voltage_reactive_power_control_mode(net, time)
            pass
        elif pv_control_mode == "power_factor_active_power_ctrl":
            # (3): 'active_reactive_power_ctrl'
            # self.active_reactive_power_control_mode(net, time)
            pass
        elif pv_control_mode == "constant_reactive_power_ctrl":
            # (4): 'constant_reactive_power_ctrl'
            # self.constant_reactive_power_control_mode(net, time)
            pass
        elif pv_control_mode == "voltage_active_power_ctrl":
            # (5): 'voltage_active_power_ctrl'
            #self.voltage_active_power_control_mode(net, time)
            pass
        elif pv_control_mode == "two_layer_local_adaptive_ctrl":
            # (6): 'two_layer_local_adaptive_controller'
            # self.two_layer_local_adaptive_ctrl(net, time)
            # Add implementation for two-layer local adaptive real-time.
            # Volt/Var Control (VVC) -> doi: 10.1109/TSG.2018.2840965
            pass
        else:
            raise ValueError("Oops! That was no valid pv control and regulation standard mode")

    def limit_pv_power_output(self):
        """Limit the active power (self.p_mw) and reactive power (self.q_mvar)
        of the PV system based on the PV Inverter limits. 
        This function ensures that the active/reactive power (p_mw/q_mvar) 
        does not exceed the maximum active power (self.max_p_mw) 
        and does not go below the minimum reactive power (self.min_q_mvar).
        """
        self.p_mw = min(self.p_mw, self.max_p_mw)
        self.q_mvar = max(self.q_mvar, self.min_q_mvar)
        # if self.gen_type == "MPV":
        #     self.q_mvar = 0.0 # due that the MPV reactive power is set to zero
        #     print(f"\n self.gen_type: {self.gen_type}, self.bus: {self.bus}")
        #     print(f"self.p_mw: {self.p_mw}, self.q_mvar: {self.q_mvar}")
        #     print(f"self.sn_mva: {np.sqrt(self.p_mw**2+self.q_mvar**2)}")
        #     print(f"self.max_p_mw: {self.max_p_mw}, self.min_p_mw: {self.min_p_mw}")
        # sn_mva = np.sqrt(self.p_mw*2+self.q_mvar*2)
        if self.p_mw == self.max_p_mw:
            print("p_mw reached predefined max.")
        if self.q_mvar == self.min_q_mvar:
            print("q_mvar reached predefined min.")
        if self.q_mvar == self.max_q_mvar:
            print("q_mvar reached predefined max.")

    def monitor_pv_control(self, time):
        """ Monitors and prints various control parameters of the PV System.
        Args:
            time (int): The current time step.
        """
        print("self.pid:", self.pid,
              "time:", time,
              "self.norm_mom_p_mw:", self.norm_mom_p_mw,
              "mom_p_mw:", self.mom_p_mw,
              "mom_cos_phi:", self.mom_cos_phi,
              "p_mw:", self.p_mw,
              "q_mvar:", self.q_mvar)

    def voltage_reactive_power_control_mode(self, net):
        """ (1): Q(V) - Voltage Reactive Power Control Mode
        Implements voltage-reactive power control (Q(V) characteristic) for PV systems.
        This method controls reactive power (Q) in response to the grid voltage (V), 
        as described by the Q(V) curve in accordance with VDE-AR-N 4105 (2018) 
        guidelines. 
        It ensures that power outputs (real and reactive) are within the technical 
        operational limits of the inverter. Depending on the normalized momentary real 
        power, it either restricts the reactive power to 10% of the maximum real power 
        as per VDE guidelines, or operates in a voltage-reactive power mode where Q is 
        determined based on the voltage level at the grid connection point.
        The function also adjusts the real power output to respect the total apparent 
        power limit, while saving unbounded reactive power values for potential 
        future usage or analysis.
        """
        # Add noise to voltage measurement
        if self.timeseries_ctrl == "control_module":
            self.add_voltage_noise(net) # self.vm_pu_meas

        # Calculate reactive power according to Q=f(V) Charasteristic curve
        q_out = self._calculate_q_out()

        # Normalize the momentary active power output to its max. value.
        self.norm_mom_p_mw = self.mom_p_mw/self.max_p_mw

        # Ensure normalized reactive power is within technical inverter range.
        self.norm_q_mvar = max(self.norm_min_q_mvar, min(q_out, self.norm_max_q_mvar))

        # Calculation normalized active power by given normalized apparent power and reactive power.
        self.norm_p_mw = min(self.norm_mom_p_mw,
                             np.sqrt(self.norm_sn_mva ** 2 - self.norm_q_mvar ** 2))
        # Calculate momentary apparent power by given normalized real power and reactive power.
        self.mom_sn_mva = np.sqrt(self.norm_p_mw ** 2 + self.norm_q_mvar ** 2)

        if self.norm_mom_p_mw <= self.reactive_power_performance_limit:
            # Store the unbounded reactive power value
            self.raw_norm_q_mvar = self.norm_q_mvar

            # Regulate reactive power to be no more than 10% of max active power, as in VDE.
            max_bounded_q_mvar = self.norm_p_mw * self.ten_prct_reactive_power_performance_limit
            q_mvar_abs = min(abs(self.norm_q_mvar), abs(max_bounded_q_mvar))

            # Update the reactive power and normalize respecting the limitation.
            self.q_mvar = np.copysign(q_mvar_abs, self.norm_q_mvar) * self.sn_mva
            self.norm_q_mvar = self.q_mvar / self.sn_mva
            # Update real power with the condition that it should not exceed the square root
            # of the difference of the square of apparent power and the square of reactive power.
            self.p_mw = min(self.mom_p_mw, np.sqrt(self.sn_mva**2 - self.q_mvar**2))
        else:
            # Store the unbounded reactive power value
            self.raw_norm_q_mvar = self.norm_q_mvar
            # The voltage-reactive power mode (Q = f(V)) determines the Q from
            # the curve based on the voltage level at the grid connection point (V_CCP).
            # The theoretically maximum possible P is calculated using this Q,
            # along with the known S, while trying to use the currently Pavailable .
            self.q_mvar = self.norm_q_mvar * self.sn_mva
            self.p_mw = min(self.mom_p_mw, np.sqrt(self.sn_mva**2 - self.q_mvar**2))

    def power_factor_active_power_ctrl(self):
        """ (2): cos φ(P) - Power Factor/Active Power Characteristic Mode
        Implements the Power Factor/Active Power Characteristic Mode as
        described in VDE-AR-N 4105 for a PV system.
        Args:
            net (pandapowerNet): The pandapower network object.
            time (float): The time instance.
        Returns:
            p_mw (float): The active power value from PV-Inverter.
            q_mvar (float): The reactive power value from PV-Inverter.
        Attributes:
            norm_mom_p_mw (float): Normalized active power values from raw data.
            mom_cos_phi (float): The desired power factor based on the load level.
            min_cos_phi (float): The minimum cosine phi.
            norm_p_mw_lower_threshold (float): The lower threshold for the
                active power level to determine the desired power factor.
            mom_sn_mva (float): The actual nominal apparent power in the PV system.
            sn_mva (float): The nominal apparent power of the PV system.
            nomi_cos_phi (float): The nominal power factor value.
        """
        # Set the current active power value
        self.norm_mom_p_mw = self.mom_p_mw/self.sn_mva
        self.norm_mom_p_mw = min(self.norm_mom_p_mw,self.norm_max_p_mw)
        # Check if p is greater than or equal to the active power threshold (50%)
        if self.norm_mom_p_mw <= self.norm_p_mw_lower_threshold:
            # Use the nominal cos_phi value(1.0)
            self.mom_cos_phi = self.nomi_cos_phi
            self.calc_mom_cos_phi = self.nomi_cos_phi
        else:
            # Calculate the reactive power required to maintain a power factor
            slope_denominator = self.norm_p_mw_lower_threshold * self.norm_max_p_mw
            slope_numerator = self.norm_mom_p_mw - slope_denominator
            slope = slope_numerator / slope_denominator
            delta_cos_phi = self.nomi_cos_phi - self.min_cos_phi
            self.calc_mom_cos_phi = self.nomi_cos_phi - slope*delta_cos_phi
        self.mom_cos_phi = max(self.min_cos_phi, self.calc_mom_cos_phi)
        self.mom_sn_mva = min(self.norm_mom_p_mw, self.norm_sn_mva)
        self.norm_p_mw, self.norm_q_mvar = self._pq_from_cos_phi(s_mva=self.mom_sn_mva,
                                                                 cos_phi=self.mom_cos_phi,
                                                                 qmode=self.qmode,
                                                                 pmode=self.pmode)
        self.p_mw = self.norm_p_mw*self.sn_mva
        self.q_mvar = self.norm_q_mvar*self.sn_mva

    def constant_power_factor_active_power_ctrl(self):
        """ (3): const cos_phi - Constant Power Factor Control Mode
        Implements the Constant Power Factor Control Mode as per 
        VDE-AR-N 4105 for a PV system.
        Args:
            cos_phi (float): The constant power factor value between 0.90 and 1.
        Returns:
            p_mw (float): The active power value from the PV-Inverter.
            q_mvar (float): The reactive power value from the PV-Inverter.
        """
        # Set the current active power value
        self.norm_mom_p_mw = self.mom_p_mw/self.sn_mva
        self.norm_mom_p_mw = min(self.norm_mom_p_mw,self.norm_max_p_mw)

        self.mom_cos_phi = min(self.min_cos_phi, self.nomi_cos_phi)
        if self.mom_cos_phi < self.min_cos_phi or self.mom_cos_phi > 1:
            raise ValueError("cos_phi must be in the range [0.90, 1]")
        self.mom_sn_mva = min(self.norm_mom_p_mw, self.norm_sn_mva)
        self.norm_p_mw, self.norm_q_mvar = self._pq_from_cos_phi(s_mva=self.mom_sn_mva,
                                                                 cos_phi=self.mom_cos_phi,
                                                                 qmode=self.qmode,
                                                                 pmode=self.pmode)
        self.p_mw = self.norm_p_mw*self.sn_mva
        self.q_mvar = self.norm_q_mvar*self.sn_mva

    def _calculate_q_out(self):
        """ Calculates reactive power output based on measured voltage. 
        This function acts as a helper for the voltage-reactive power control
        mode (Q(V) characteristic).
        Return:
            q_out (float): Reactive power output based on the Q(V) characteristic. 
        """
        # Check voltage measurements and calculate reactive power accordingly
        if self.vm_pu_meas <= self.v_1:
            q_out = self.norm_max_q_mvar
        elif self.vm_pu_meas <= self.v_2:
            m_12 = (self.nomi_q_mvar - self.norm_max_q_mvar) / (self.v_2 - self.v_1)
            v_dev = self.vm_pu_meas - self.v_1
            q_out = m_12 * v_dev + self.norm_max_q_mvar
        elif self.vm_pu_meas <= self.v_4:
            q_out = self.nomi_q_mvar
        elif self.vm_pu_meas <= self.v_5:
            m_45 = (self.norm_min_q_mvar - self.nomi_q_mvar) / (self.v_5 - self.v_4)
            v_dev = self.vm_pu_meas - self.v_4
            q_out = m_45 * v_dev + self.nomi_q_mvar
        else:
            q_out = self.norm_min_q_mvar
        return q_out

    def _pq_from_cos_phi(self, s_mva, cos_phi, qmode, pmode):
        """ Calculates active and reactive power from power factor."""
        # Get the signs for reactive and active power modes
        p_sign = self.get_p_sign(pmode)
        q_sign = self.get_q_sign(qmode)
        # Calculate active power
        p_mw = s_mva * cos_phi
        # Calculate reactive power
        q_mvar = p_sign * q_sign * np.sqrt(s_mva ** 2 - p_mw ** 2)
        return p_mw, q_mvar