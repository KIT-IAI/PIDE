"""Storage Control module."""

import os
import numpy as np
import pandapower as pp
from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import pandapowerNet

# from pandapower.timeseries.data_sources.frame_data import DFData
from typing import Optional, Union, Dict, Any


class StorageController(Controller):
    """
    This class represents a Storage Controller in a power network simulation.
    It simulates a time step in a time-series simulation for the battery,
    updating its state of charge (SoC) and power values from a specified profile
    or algorithm.
    INPUT:
        **net** (attrdict) - Pandapower net
        **sid** (int) - ID of the storage unit that is controlled
        **storage_p_control_mode** (str, None) - Selected algorithm in config
        **storage_q_control_mode** (str, None) - Selected algorithm in config
        **data_source** (str) - Data source for the momentary active power
                                **p_mw** (float)
                                (positive for charging, negative for discharging)
        **profile_name** (str[]) - Power profile for the control in data_source
    OPTIONAL:
        **resolution** (int, 15) - Resolution of the control in 15 minutes.
        **scale_factor** (float, 1.0) - Resolution of the control in 15 minutes.
        **initial_soc** (float, 50.0) - The initial state of charge of the storage.
        **order** (int, 0) - The order of the controller in the overall control process.
        **level** (int, sid) - The priority level of the controller.
        **in_service** (bool) - Indicates if the element is currently active
        **recycle** (bool, True) - Re-use of internal-data in a time series loop.
        Settings can be made in the configuration file
        Optional to be set in advance.
    Methods:
        time_step(net, time):
        Simulates a time step in a time-series simulation for the battery,
        updating its state of charge and power values.
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

    def __init__(
        self,
        net: pandapowerNet,  # replace with the actual type
        sid: int,
        storage_p_control_mode: str,
        storage_q_control_mode: str,
        regulation_standard: str,
        timeseries_ctrl: str,
        inverter_ctrl_params: Dict[str, Any],
        output_data_path: str,
        data_source: Optional[str] = None,
        profile_name: int = 0,
        resolution: int = 15,
        inital_soc: float = 12.0,
        mcs_settings=Dict[int, Any],
        scale_factor: float = 1.0,
        order: int = 2,
        level: int = 0,
        test_folder_name: str = "storage_controller_tests",
        in_service: bool = True,
        recycle: bool = True,
        **kwargs: Any,
    ):
        # Call the parent's(Controller) init function
        super().__init__(
            net,
            in_service=in_service,
            recycle=recycle,
            order=order,
            level=level,
            initial_powerflow=True,
            **kwargs,
        )
        self.der_element_type = "storage"
        self.timeseries_ctrl = timeseries_ctrl
        # Modularize the initialization into separate methods
        self.init_seeds_and_error_params(mcs_settings=mcs_settings)
        self.init_voltage_measurements()
        self.init_constants(scale_factor=scale_factor)
        self.read_storage_attributes(net=net, did=sid)
        self.init_power_parameters()
        self.init_storage_parameters(net=net, inital_soc=inital_soc)
        self.init_der_control_mode_settings(
            net=net,
            der_control_mode=storage_p_control_mode,
            storage_q_control_mode=storage_q_control_mode,
            regulation_standard=regulation_standard,
            inverter_ctrl_params=inverter_ctrl_params,
        )
        self.configure_output_path(
            output_data_path=output_data_path, folder_name=test_folder_name
        )
        # Load PF(power factor), Regulation and Control Settings
        self.load_power_factor_settings()
        self.load_control_rule_settings(net=net, element_type=self.der_element_type)
        self.load_ctrl_mode_and_technical_params()
        # Time Series Data Settings
        self.init_time_steps(resolution=resolution)
        self.init_data_source(
            net=net, data_source=data_source, profile_name=profile_name
        )

    def init_constants(self, scale_factor):
        """Initializes the constants to be used in this class.
        Args:
            scale_factor (float): The factor by which to scale the input values.
        """
        # Define constants for the number of decimal places
        self.c_axis_factor = 1.1
        # Define constants for the number of decimal places
        self.c_precision = 4
        # Scaling factor for input values
        self.scale_factor = scale_factor

    def read_storage_attributes(self, net, did):
        """Reads the PV attributes from the network.
        Args:
            net: The network object.
            did: The id of the DER unit.
            sid: The id of the Storage unit.
        """
        # Assign the Storage attributes from the network
        self.sid = did  # id of controlled Storage unit
        # Bus id where the Storage system is connected
        self.bus = net.storage.at[did, "bus"]
        # Bus name where the Storage system is connected
        self.bus_name = net.bus.at[self.bus, "name"]
        # Storage name
        self.name = net.storage.at[did, "name"]

        # Active power of the storage (positive for charging!,negative for discharging!)
        self.p_mw = net.storage.at[did, "p_mw"]
        # Reactive power of the Storage system
        self.q_mvar = net.storage.at[did, "q_mvar"]
        # Nominal (apparent) power of the Storage system
        self.sn_mva = net.storage.at[did, "sn_mva"]
        self.rated_apparent_power = net.storage.at[did, "sn_mva"]
        self.index = net.storage.index[did]
        self.gen_type = net.storage.at[did, "type"]
        self.in_service = net.storage.at[did, "in_service"]
        self.zone = net.storage.at[did, "zone"]

    def init_power_parameters(self):
        """Initializes the power parameters of the Bus unit."""
        # Storage-Power Parameters VDE-AR-N 4105 or the IEEE Std 1547-2018 Settings
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
        self.ten_prct_reactive_power_performance_limit = 0.10
        self.reactive_power_performance_limit = 0.10

    def init_voltage_measurements(self):
        """Initializes the Bus id voltage measurements where the Storage system
        is connected."""
        # Add Voltage Measurements
        self.add_vm_pu_noise = True
        self.update_meas = True

    def init_der_control_mode_settings(
        self,
        net,
        der_control_mode,
        storage_q_control_mode,
        regulation_standard,
        inverter_ctrl_params,
    ):
        """Initializes the control mode of the Storage unit.
        Args:
            net: The network object.
            storage_q_control_mode: The control mode of the PV unit.
            regulation_standard: The regulation standard for the PV system.
            inverter_ctrl_params: The control parameters for the inverter.
        """
        # Running a power flow analysis
        pp.runpp(net)
        # Network object
        self.net = net
        # Base Inverter Params
        self.base_inverter_ctrl_params = inverter_ctrl_params
        # Control mode of the DER Storage systems
        self.storage_p_control_mode = der_control_mode
        # Q-Control mode of the DER
        self.storage_q_control_mode = storage_q_control_mode
        # Regulation standard for DER systems
        self.regulation_standard = regulation_standard
        # Inverter control mode parameters
        self.inverter_ctrl_params = inverter_ctrl_params[regulation_standard]
        # Regulation standard mode parameters
        self.regulation_standard_mode = inverter_ctrl_params[regulation_standard][
            "standard_mode"
        ]
        # Selected Inverter Control and Criticality parameters
        self.selected_inverter_ctrl_params = self.inverter_ctrl_params[
            self.regulation_standard_mode
        ]
        self.criticality_params = inverter_ctrl_params["criticality"]
        self.applied = False

    def configure_output_path(self, output_data_path, folder_name):
        """Configures the output paths.
        Args:
            output_data_path: The base path for the output data.
        """
        # Prepare output path
        self.output_data_path = output_data_path
        self.output_elm_control_path = os.path.join(output_data_path, folder_name)

    def init_time_steps(self, resolution):
        """Initializes the time steps.
        Args:
            resolution: The resolution of the time steps.
        """
        # TimeSteps settings(default: 15min)
        self.resolution = resolution
        # Time constants "c" settings
        self.c_hours_in_mins = 60  # default: 15min values
        self.c_days_in_hours = 24  # default: 15min values
        self.steps_per_hour = self.resolution / self.c_hours_in_mins

    def init_data_source(self, net, data_source, profile_name):
        """Initializes the data source for the PV unit.
        Args:
            net: The network object.
            data_source: The data source for the time series values.
            profile_name: The name of the profile in the data source.
        """
        # Data source for time series values
        self.data_source = data_source
        self.profile_name = profile_name
        # Scaling for the Storage unit
        self.scaling = net.storage.at[self.sid, "scaling"]
        # Initialize the last time step as None
        self.last_time_step = None

    def init_storage_parameters(self, net, inital_soc):
        """Initializes the parameters related to the state of charge (SoC) of
        the Storage system.
        Args:
            net: The network object.
        """
        # Current energy capacity of the storage unit in MWh (self.e_mwh)
        self.current_energy_mwh = None
        self.inital_soc = inital_soc
        self.soc_percent = net.storage.at[self.sid, "soc_percent"] = self.inital_soc
        self.soc_percent_discharge_limit = 24.00
        self.soc_percent_charge_limit = 86.0
        self.max_soc_percent = 99.99
        self.min_soc_percent = 0.01
        self.discharge_power_loss = 0.00
        self.storage_self_discharge_time_step = None
        self.set_efficiency_and_self_discharge(net, self.sid)

    def init_seeds_and_error_params(self, mcs_settings):
        """Sets the random seed for each type of modelling error defined in the
        configuration and prints the seed values."""
        self.mcs_settings = mcs_settings
        self.seed_value = mcs_settings["seed_value"]
        self.loc = mcs_settings["loc"]
        self.error_std_dev = mcs_settings["error_std_dev"]
        # Setzen Sie den Seed (z.B. auf den Wert 42)
        np.random.seed(self.seed_value)

    def get_initial_value(self, net, elm_name, sid, param_name, default_val):
        """Returns initial value from net storage or default_val if param not
        present."""
        if param_name in net[elm_name].keys():
            return net[elm_name].at[sid, param_name]
        return default_val

    def generate_error(self, value):
        """Generates error based on the provided seed_key."""
        if self.add_vm_pu_noise:
            error = np.random.normal(loc=self.loc, scale=self.error_std_dev)
            return value * error
        else:
            error = 0
            return value * error

    def calculate_param_with_error(self, net, elm_name, sid, param_name, default_val):
        """Sets parameter value considering possible errors and fluctuations."""
        initial_value = self.get_initial_value(
            net, elm_name, sid, param_name, default_val
        )
        add_modelling_error = self.generate_error(value=initial_value)
        return initial_value + add_modelling_error

    def set_efficiency_and_self_discharge(self, net, sid):
        """Sets the efficiency and self-discharge rates of the storage unit,
        taking into account possible errors and fluctuations in these parameters.
        Parameters:
        net (pandas.DataFrame): The dataframe containing information about the
                                storage units in the network.
        sid (int): The ID of the storage unit for which the parameters are being set.
        """
        self.efficiency_percent = self.calculate_param_with_error(
            net=net,
            elm_name="storage",
            sid=sid,
            param_name="efficiency_percent",
            default_val=0.95,
        )
        self.self_discharge_percent_per_day = self.calculate_param_with_error(
            net=net,
            elm_name="storage",
            sid=sid,
            param_name="self-discharge_percent_per_day",
            default_val=0.13,
        )

    def add_voltage_noise(self, net, voltage_error_std_dev=0.001):
        """Adds Gaussian noise to the voltage measurement of a bus in a power grid.
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
        self.vm_pu = net.res_bus.at[self.bus, "vm_pu"]  # self.v_pcc
        if self.add_vm_pu_noise:
            # Generate Gaussian noise with loc_X=self.loc and X_std=self.error_std_dev:
            self.voltage_noise = np.random.normal(loc=0, scale=voltage_error_std_dev)
            # Apply error to the voltage measurement
            self.vm_pu_meas = self.vm_pu + self.voltage_noise
        else:
            self.vm_pu_meas = self.vm_pu

    def load_power_factor_settings(self, test_mode=False):
        """
        Set cos_phi values (underexcited & overexcited) based on the rated apparent
        power of Storage, following the regulation standards for Storages (PVs, ESS, EVCS).
        The regulations can be either German VDE Technical Low Voltage Regulations
        (VDE-AR-N 4105) or IEEE Std 1547-2018.
        This function checks if the rated apparent power is <= 0.0046 MVA (4.6 kVA),
        and sets the underexcited & overexcited cos_phi values accordingly.
        Storages-Unit sn_mva <= 0.0046 MVA (4.6 kVA):
            For Storages with a rated apparent power of sn_mva <= 0.0046 MVA,
            the grid operator provides a fixed cos_phi value between
            cos_phi = 0.95 lagging and cos_phi = 0.95 leading.
        Storages-Unit sn_mva > 0.0046 MVA (4.6 kVA):
            For Storages with a rated apparent power of sn_mva > 0.0046 MVA,
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
        # Check Storage for setting cos_phi values
        if self.regulation_standard.lower() == "vde":
            if test_mode or self.sn_mva <= 0.0046:
                # Settings for Storages <= 4.6 kVA
                self.min_cos_phi = 0.95
                self.available_reactive_power = self.calc_allowed_reactive_power(
                    self.min_cos_phi
                )
                # If 0 ≤ P / PEmax < 0.2, then DER terminals Qmax ≤ 0.2 * PEmax
                self.reactive_power_performance_limit = 0.200
            else:
                # Settings for Storages > 4.6 kVA or in test mode
                self.min_cos_phi = 0.900
                self.available_reactive_power = self.calc_allowed_reactive_power(
                    self.min_cos_phi
                )
                # If 0 ≤ P / PEmax < 0.1, then DER terminals Qmax ≤ 0.1 * PEmax
                self.reactive_power_performance_limit = 0.100
        elif self.regulation_standard.lower() == "ieee":
            # Add implementation for IEEE Std 1547-2018 here
            pass
        else:
            raise ValueError(
                "Unsupported standard. Supported standards are"
                " 'VDE-AR-N 4105' and 'IEEE Std 1547-2018'."
            )

    def load_ctrl_mode_and_technical_params(self):
        """Sets selected standard mode parameters for the inverter."""
        # Define control and criticality(both voltage and grid) parameter keys.
        self.ctrl_mode_param_keys_1 = [
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
        Notes:
            Read controllable element attributes from net - necessary for OPF
        Raises:
            ValueError: If an unsupported element type is provided.
        """
        if element_type not in ["sgen", "storage"]:
            raise ValueError(
                f"Unsupported element type: {element_type}."
                "Supported types: 'sgen', 'storage'."
            )
        # Load general distributed energy resources (DER) settings
        self._load_der_settings()
        # Load settings for Storage system
        self._load_storage_settings(net)
        # Load settings for voltage magnitude
        self._load_voltage_magnitude_settings(net)

    def _load_der_settings(self):
        """Load general Distributed Energy Resource (DER) settings."""
        # Minimum active power
        self.min_p_mw = 0  # Set to zero since it's a PV system
        # Maximum active power
        self.max_p_mw = self.sn_mva
        # Momentary acitve, reactive and apparent power
        self.mom_p_mw = 0  # raw since unprocessed or measured size
        self.mom_q_mvar = 0
        self.mom_sn_mva = 0
        self.mom_bat_p_mw = 0
        self.mom_bat_q_mvar = 0
        # Norm Maximum/Minimum reactive power (Q) injection/absorption
        self.norm_min_q_mvar = -self.available_reactive_power
        self.norm_max_q_mvar = +self.available_reactive_power
        self.min_q_mvar = self.norm_min_q_mvar * self.sn_mva
        self.max_q_mvar = self.norm_max_q_mvar * self.sn_mva
        self.nomi_q_mvar = 0
        self.norm_max_p_mw = self.max_p_mw / self.max_p_mw
        self.norm_min_p_mw = 0
        self.norm_sn_mva = self.sn_mva / self.sn_mva
        self.norm_p_mw = None
        self.norm_q_mvar = None
        self.norm_mom_p_mw = 0.0
        self.norm_mom_q_mvar = None
        self.norm_mom_sn_mva = None

        # Save the unbounded reactive power for voltage control
        self.raw_norm_q_mvar = 0
        # Define active power thresholds for cosinus phi control per VDE/IEEE rules.
        self.norm_p_mw_lower_threshold = 0.5  # 50.0
        self.norm_p_mw_upper_threshold = 1.0  # 100.0

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
        """Load voltage magnitude settings from the network, defaulting to
        predefined limits.
        Args:
            net (pandapowerNet): Pandapower network data structure.
        """
        # Maximum and Minimum voltage magnitude per-unit (PU)
        # Use value from net if available, else use upper voltage deviation
        self.max_vm_pu = (
            net.bus.at[self.bus, "max_vm_pu"]
            if not np.isnan(net.bus.at[self.bus, "max_vm_pu"])
            else self.upper_vm_pu
        )
        # Use value from net if available, else use lower voltage deviation
        self.min_vm_pu = (
            net.bus.at[self.bus, "min_vm_pu"]
            if not np.isnan(net.bus.at[self.bus, "min_vm_pu"])
            else self.lower_vm_pu
        )

    def _load_storage_settings(self, net):
        """Load control settings specific for the 'storage' elements."""
        # Storage specific code
        # The maximum energy content of the storage (maximum charge level)
        self.max_e_mwh = net.storage.at[self.sid, "max_e_mwh"] * self.scale_factor
        net.storage.at[self.sid, "max_e_mwh"] = self.max_e_mwh
        # Minimum(discharging)/Maximum(charging) active power
        self.min_p_mw = (
            net.storage.at[self.sid, "min_p_mw"]
            if "min_p_mw" in net.storage.keys()
            else -self.max_e_mwh / 5
        )
        self.max_p_mw = (
            net.storage.at[self.sid, "sn_mva"]
            if "sn_mva" in net.storage.keys()
            else self.max_e_mwh / 5
        )
        # Maximum/Minimum reactive power (Q) injection/absorption
        # Minimum (discharging) reactive power
        self.min_q_mvar = (
            net.storage.at[self.sid, "min_q_mvar"]
            if "min_q_mvar" in net.storage.keys()
            else self.min_q_mvar
        )
        # Maximum (charging) reactive power
        self.max_q_mvar = (
            net.storage.at[self.sid, "max_q_mvar"]
            if "max_q_mvar" in net.storage.keys()
            else self.max_q_mvar
        )
        # Overwrite inverter (sid) values following VDE guidelines
        net.storage.at[self.sid, "max_q_mvar"] = self.max_q_mvar
        net.storage.at[self.sid, "min_q_mvar"] = self.min_q_mvar
        net.storage.at[self.sid, "max_p_mw"] = self.max_p_mw
        net.storage.at[self.sid, "min_p_mw"] = self.min_p_mw

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
        """Calculates the P-Sign value based on the given P mode.
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
        """Calculates the Q-Sign value based on the given Q mode.
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
                f' or "overexcited" (Q injection, increases voltage)'
            )
        return qsign

    def _check_reactive_power_tolerance(self, tolerance=0.02):
        """Check if the calculated reactive power is within the specified
        tolerance range of max_q_mvar. If the calculated reactive power is
        outside the range, a message will be printed indicating that it is
        outside the tolerance.
        Args:
            tolerance (float, optional): The allowed percentage difference.
            Defaults to 0.02 (2%)."""
        # Calculate reactive power
        calc_q_mvar = round(
            (self.norm_sn_mva**2 - self.norm_mom_p_mw**2) ** 0.5, self.c_precision
        )
        # Calculate the lower & upper limit for acceptable reactive power
        lower_limit = self.norm_max_q_mvar * (1 - tolerance)
        upper_limit = self.norm_max_q_mvar * (1 + tolerance)
        if not lower_limit <= calc_q_mvar <= upper_limit:
            print("calc_q_mvar is outside the 2% tolerance of max_q_mvar.")

    def get_stored_energy(self):
        """
        Calculates the amount of energy currently stored in the storage unit.
        Returns:
            stored_energy (float):
                The amount of energy stored in the unit, in MWh.
        Notes:
            The stored energy is calculated as the product of the maximum energy capacity
            of the storage unit, in MWh (self.max_e_mwh), and the current state of charge
            of the unit, expressed as a percentage (self.soc_percent). The result is divided
            by 100 to convert the percentage to a decimal value.
        """
        return self.max_e_mwh * self.soc_percent / 100

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
            This function checks if the 'applied' attribute of the controller object
            is True, indicating that the controller has already been applied. It also
            checks if the power flow solution in the network has converged using
            the 'ppc' attribute of the pandapower network object. If the power flow
            solution has not converged, this function will return False even
            if the controller has been applied.
        """
        # return self.applied
        return self.applied and net["_ppc"]["success"]

    def write_to_net(self, net):
        """
        Writes the active and reactive power output and state of charge of the
        storage unit to the pandapower network object.
        Args:
            net (pandapowerNet): The pandapower network object.
        Notes:
            This function updates the 'p_mw', 'q_mvar', and 'soc_percent'
            attributes of the storage unit with the controller's current values.
            The 'sid' attribute of the controller object is used to locate the
            corresponding storage unit in the network. The changes are made
            directly to the 'storage' dataframe within the network object.
        """
        if (
            self.soc_percent >= self.soc_percent_discharge_limit
            and self.soc_percent <= self.soc_percent_charge_limit
        ):
            # Write updated p_mw and q_mvar to the pandapower net
            net.storage.at[self.sid, "p_mw"] = self.p_mw
            net.storage.at[self.sid, "q_mvar"] = self.q_mvar

        else:
            # Write updated p_mw and q_mvar to the pandapower net
            net.storage.at[self.sid, "p_mw"] = 0
            net.storage.at[self.sid, "q_mvar"] = 0
        # Write updated SoC to the pandapower net
        net.storage.at[self.sid, "soc_percent"] = self.soc_percent
        net.storage.at[self.sid, "discharge_power_loss"] = self.discharge_power_loss

    def control_step(self, net):
        """
        Executes a control step by writing the current control values to the
        network object.
        Args:
            net (pandapowerNet): The pandapower network object.
        Notes:
            This function updates the active and reactive power output and state
            of charge of the storage unit in the network object by calling the
            'write to net' function. It also sets the 'applied' attribute of the
            controller object to True, indicating that the values set in the
            previous control step have been included in the load flow calculation.
            This is useful for determining whether the controller has already
            been applied to the network in subsequent control steps.
        """
        self.write_to_net(net)
        self.applied = True

    def _reset_applied(self):
        """Resets the applied variable to False."""
        self.applied = False

    def time_step(self, net, time):
        """This function Simulates a time step in a time-series simulation for
        the battery, updating its new state of (SoC), active power (p_mw) and
        reactive power (q_mvar) values from a profile or algorithm.
        Args:
            - net (Pandapower net object): Pandapower net object
            - time (int): Current time in the simulation.
        Behavior:
            - Updates the state of charge (SoC) based on the battery's power
              values and the maximum stored energy on the previous time step.
            - Updates the SoC based on the self-discharge of the battery and
              writes the updated SoC to the pandapower net.
            - If the battery has a data source, runs the battery control and
              monitoring functions based on the specified control mode.
        """
        self._reset_applied()
        if self.last_time_step is None:
            self.net.sgen.p_mw = 0.0
        if self.last_time_step is not None:
            # Update Storage SoC (state of charge) based on previous time step (last time step)
            self.set_efficiency_and_self_discharge(net, self.sid)
            self.update_soc_from_previous(time)
            self.update_soc_from_self_discharge()
        self.last_time_step = time
        # Read new values from a data source and profile
        if time == 40:
            a = 1
        # Get/Read new values of the storage element from data source
        if self.data_source:
            if self.profile_name is not None:
                # Get active power values for current time step
                self.read_active_power_values(time)
                # Ihre Logik für Lade-/Entladeentscheidungen
                self.run_battery_p_control(
                    net=net,
                    time=time,
                    storage_p_control_mode=self.storage_p_control_mode,
                )
                self.update_and_monitor_storage_power(
                    desired_p_mw=self.mom_bat_p_mw, time=time
                )
                # Run battery control based on the selected Q-Control mode
                self.run_battery_q_control(
                    net=net, storage_q_control_mode=self.storage_q_control_mode
                )
            else:
                pass
        else:
            raise ValueError("Invalid Storage Control Mode specified")
        # Writes p_mw, q_mvar and soc_percent of the storage unit to the net
        # self.write_to_net(net)

    def run_battery_p_control(self, net, time, storage_p_control_mode):
        """The `run_battery_p_control` function determines the battery's
        charging or discharging status based on the selected storage control
        mode.
        Supports five different storage control modes:
        1. Raw-databased Control (RDC) of BES:
            - Charging or discharging status based on raw data.
        2. Rule-Based Control (RBC) for Decentralized Solar-Powered BES:
            - Charging or discharging rate/status determined by decentralized
              control of solar energy in low-voltage grids.
        3. Rule-Based Control (RBC) for Distributed BES:
            - Manages power based on total consumption and PV generation in a
              segmented control zone.
        4. Rule-Based Control (RBC) for Distributed BES with Day/Night Cycle (DNC) Controlling:
            - Combines time-based charging/discharging cycles with distributed
              solar energy control.
        5. Rule-Based Control (RBC) for Decentralized BES with Day/Night Cycle (DNC) Controlling (local):
            - Focuses on time-aligned charging and discharging cycles dependent
              on the time of day.
        Raises a ValueError if the selected storage control mode is not recognized.
        INPUT:
           **net** (pandapowerNet) - The Pandapower network
           **time** (int) - Current time step of the simulation
           **storage_p_control_mode** (str) - Selected storage control mode
        OUTPUT:
           **p_mw** (float) - Calculated power output based on the selected
                              storage control mode
        """
        # Functionality for different storage control modes is implemented here
        if storage_p_control_mode == "datasource":
            # (1) Raw-databased Control(RDC) of Energy Storage Unit:
            p_mw = self.raw_databased_control()
        elif storage_p_control_mode == "rbc_pvbes_decentralized_sc_ctrl":
            # (2) Rule-Based Control (RBC) for Decentralized PV-Powered Battery Energy Storage (BES)
            p_mw = self.rbc_pvbes_decentralized_sc_ctrl(net)
        elif storage_p_control_mode == "rbc_pvbes_distributed_sc_ctrl":
            # (3) RBC for Distributed BES
            p_mw = self.rbc_pvbes_distributed_sc_ctrl(net)
        elif storage_p_control_mode == "rbc_pvbes_distributed_sc_dnc_ctrl":
            # (4) RBC for Distributed BES with DNC
            p_mw = self.rbc_pvbes_distributed_sc_dnc_ctrl(net, time)
        elif storage_p_control_mode == "rbc_bes_dnc_ctrl":
            # (5) RBC for Decentralized BES with DNC (local)
            p_mw = self.rbc_bes_dnc_strategy_ctrl(net, time)
        else:
            raise ValueError("Oops! That was no valid storage control mode")
        # Return the calculated power (p_mw) as the momentary battery power output
        self.mom_bat_p_mw = p_mw

    def update_and_monitor_storage_power(self, desired_p_mw, time):
        """This function controls and updates p_mw of the storage system based on SoC,
        and monitors the storage control if the power level is above a certain threshold.
        INPUT:
           **desired_p_mw** (float) - Desired power output
           **time** (int) - Current time step of the simulation
        """
        self.mom_bat_p_mw = self.adjust_storage_power_output(desired_p_mw=desired_p_mw)
        # Monitor Storage control if the power level is above 50% of the specified value
        if self.norm_mom_p_mw > self.norm_p_mw_lower_threshold:
            self.monitor_storage_control(time=time, mom_bat_p_mw=self.mom_bat_p_mw)
        if self.q_mvar > 0:
            self.monitor_storage_control(time=time, mom_bat_p_mw=self.mom_bat_p_mw)

    def read_active_power_values(self, time):
        """Read active power values from a raw data profile of the storage element
        and set reactive power to zero.
        Args:
            time (int): The current time step.
        """
        self.mom_p_mw = self.data_source.get_time_step_value(
            time_step=time,
            profile_name=self.profile_name,
            scale_factor=self.scale_factor,
        )
        self.mom_q_mvar = 0
        self.norm_mom_p_mw = self.mom_p_mw / self.sn_mva
        self.norm_mom_q_mvar = 0

    def update_soc_from_previous(self, time):
        """Updates the state of charge based on the previous time step."""
        # Calculating energy change in the last time step
        delta_time = time - self.last_time_step
        delta_energy = self.p_mw * self.efficiency_percent * delta_time
        # Current energy capacity of the storage unit in MWh (self.e_mwh)
        self.current_energy_mwh = delta_energy * self.steps_per_hour
        # delta_energy *= (self.resolution / self.c_hours_in_mins)
        # Update the state of charge (SoC) based on the delta energy and
        # the maximum stored energy
        self.soc_percent += (self.current_energy_mwh / self.max_e_mwh) * 100
        # Ensure soc_percent does not exceed 100%
        self.soc_percent = min(self.soc_percent, self.max_soc_percent)
        self.soc_percent = max(self.soc_percent, self.min_soc_percent)

    def update_soc_from_self_discharge(self):
        """Updates the state of charge based on the self-discharge of the storage."""
        # Calculate the percentage self-discharge of the storage per time step
        self.storage_self_discharge_time_step = (
            self.self_discharge_percent_per_day / self.c_days_in_hours
        ) * (self.resolution / self.c_hours_in_mins)
        # self-discharge power loss in mwh
        self.discharge_power_loss = (
            self.storage_self_discharge_time_step * self.max_e_mwh
        )
        # Update the SoC based from self-discharge power loss multiplied with 100 to get soc_percent
        self.soc_percent -= self.discharge_power_loss * 100

    def adjust_storage_power_output(self, desired_p_mw):
        """Adjust the power output of the storage system based on the current state
        of charge (soc). The desired power output in MW cannot exceed the maximum
        power (self.max_p_mw) or go below the minimum power (self.min_p_mw).
        Args:
            desired_p_mw (float): The desired power output in MW.
        """
        if (self.soc_percent < self.soc_percent_charge_limit) and (desired_p_mw > 0):
            # positive p_mw (charging)
            mom_bat_p_mw = min(desired_p_mw, self.max_p_mw)
        elif (self.soc_percent > self.soc_percent_discharge_limit) and (
            desired_p_mw <= 0
        ):
            # negative p_mw (discharging)
            mom_bat_p_mw = max(desired_p_mw, self.min_p_mw)
        else:
            mom_bat_p_mw = 0
        return mom_bat_p_mw

    def monitor_storage_control(self, time, mom_bat_p_mw):
        """A function for monitoring battery storage control"""
        # Run power flow calculation using the pandapower library
        # pp.runpp(net)
        # Get the voltage at the point of common coupling (PCC)
        # v_pcc = net.res_bus.at[self.bus, "vm_pu"]
        # Check if the voltage is within the acceptable range
        # v_flag = [self.v_lower <= v_pcc <= self.v_upper]
        # Print information about the current state of the battery storage
        # p_res_mw negativ discharging
        # Print information about the voltage and voltage deviation at the PCC
        # print("v_flag:",v_flag,"v_pcc:",v_pcc, "delta_v_pcc:", v_pcc-self.v_pcc)
        # self.v_pcc = v_pcc
        print(
            "self.sid:",
            self.sid,
            " time:",
            time,
            "p_mw:",
            self.p_mw,
            "mom_bat_p_mw:",
            mom_bat_p_mw,
            "self.sid:",
            self.sid,
            "soc_percent:",
            self.soc_percent,
        )
        # Check if the battery storage is discharging more power than the load
        if mom_bat_p_mw > 0:
            print("res p_mw is greater as load")
        # (3) Check and Update the SoC if necessary
        if (self.soc_percent >= self.soc_percent_charge_limit) and (mom_bat_p_mw > 0):
            print("State of charge (soc) 90 percent reached!")
        elif (self.soc_percent <= self.soc_percent_discharge_limit) and (
            mom_bat_p_mw < 0
        ):
            print("State of charge (soc) cannot drop below 20 percent!")
        else:
            pass

    def run_battery_q_control(self, net, storage_q_control_mode):
        """
        The function determines the amount of reactive power Q for the storage
        unit based on the selected storage: 'storage_q_control_mode'.
        This function supports two different Q control modes
        [VDE-AR-N 4105 (2018)]:
            1. Voltage Reactive Power Control for Battery DER:
               The charging or discharging status is based on voltage reactive
               power control.
            2. Constant Power Factor Active Power Control for Battery DER:
               The charging or discharging rate/status is determined based on a
               constant power factor active power control.
        If the selected Q control mode is not recognized, the function raises
        a ValueError with an appropriate message.
        INPUT:
           **net** (pandapowerNet) - The Pandapower network
           **time** (int) - Current time step of the simulation
           **storage_q_control_mode** (str) - Selected Q control mode
        """
        if storage_q_control_mode == "voltage_reactive_power_ctrl":
            # (1):'voltage_reactive_power_ctrl' for Battery DER
            self.voltage_reactive_power_control_mode(
                net=net, mom_p_mw=self.mom_bat_p_mw
            )
        elif storage_q_control_mode == "constant_power_factor_active_power_ctrl":
            # (2): 'constant_power_factor_active_power_ctrl' for Battery DER
            self.constant_power_factor_active_power_ctrl(mom_p_mw=self.mom_bat_p_mw)
        elif storage_q_control_mode == "datasource":
            # (1):'datasource' for Battery DER
            self.raw_databased_control()
        else:
            raise ValueError("Oops! That was no valid storage control mode")

    def raw_databased_control(self):
        """Sets the active power (p_mw) and reactive power (q_mvar) based on the
        instantaneous power values obtained from the raw data control strategy.
        """
        self.p_mw = self.mom_bat_p_mw
        self.q_mvar = 0
        return self.mom_bat_p_mw

    def voltage_reactive_power_control_mode(self, net, mom_p_mw):
        """(1): Q(V) - Voltage Reactive Power Control Mode
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
        Low Power Range Condition:
            For the range 0 ≤ Pmom/PEmax < 0.2, the reactive power should not exceed
            10%/20% of the maximum active power.
        Args:
            net -- egrid
            mom_p_mw -- momentary active power desired by run_battery_p_control
        Returns:
            self.p_mw -- active power calculated by reactive power control for DER-Inverter
            self.q_mvar -- reactive power calculated by reactive power control for DER-Inverter
        """
        # Add noise to voltage measurement
        if self.timeseries_ctrl == "control_module":
            self.add_voltage_noise(net)  # self.vm_pu_meas

        # Calculate reactive power according to Q=f(V) Charasteristic curve
        q_out = self._calculate_q_out()
        # Normalize the momentary active power output to its max. value.
        self.norm_mom_p_mw = mom_p_mw / self.max_p_mw
        # Ensure normalized reactive power is within technical inverter range.
        self.norm_q_mvar = max(self.norm_min_q_mvar, min(q_out, self.norm_max_q_mvar))
        # Calculation normalized active power by given normalized apparent power and reactive power.
        self.norm_p_mw = min(
            self.norm_mom_p_mw, np.sqrt(self.norm_sn_mva**2 - self.norm_q_mvar**2)
        )
        # Calculate momentary apparent power by given normalized real power and reactive power.
        self.mom_sn_mva = np.sqrt(self.norm_p_mw**2 + self.norm_q_mvar**2)
        # Check Low Power Range Condition
        if np.abs(self.norm_mom_p_mw) <= self.reactive_power_performance_limit:
            # Store the unbounded reactive power value
            self.raw_norm_q_mvar = self.norm_q_mvar

            # Regulate reactive power to be no more than 10% of max active power, as in VDE.
            max_bounded_q_mvar = (
                self.norm_p_mw * self.ten_prct_reactive_power_performance_limit
            )
            q_mvar_abs = min(abs(self.norm_q_mvar), abs(max_bounded_q_mvar))

            # Update the reactive power and normalize respecting the limitation.
            self.q_mvar = np.copysign(q_mvar_abs, self.norm_q_mvar) * self.sn_mva
            self.norm_q_mvar = self.q_mvar / self.sn_mva

            # Update real power with the condition that it should not exceed the square root
            # of the difference of the square of apparent power and the square of reactive power.
            self.p_mw = min(mom_p_mw, np.sqrt(self.sn_mva**2 - self.q_mvar**2))
        else:
            # Active power exceeds 10% or 20% of the maximum active power.
            # Store the unbounded reactive power value
            self.raw_norm_q_mvar = self.norm_q_mvar
            # Q is determined by voltage (V_CCP) in voltage-reactive power mode (Q = f(V)).
            # Calculate max possible P using this Q and known S, considering available P.
            self.q_mvar = self.norm_q_mvar * self.sn_mva
            self.p_mw = min(mom_p_mw, np.sqrt(self.sn_mva**2 - self.q_mvar**2))

    def constant_power_factor_active_power_ctrl(self, mom_p_mw):
        """(2): const_cos_phi - Constant Power Factor Control Mode
        Implements the Constant Power Factor Control Mode as per
        VDE-AR-N 4105 for a PV system.
        After running load_power_factor_settings function, the cos_phi value
        is set as follows:
            - (1): Q(U) - Voltage Reactive Power Control Mode
            with (adjustment range between cos φ = 0.90 lagging and cos φ = 0.90 leading)
            - (2): Constant Power Factor Control Mode
            (between cos φ = 0.90 lagging and cos φ = 0.90 leading)
        Low Power Range Condition:
            For the range 0 ≤ Pmom/PEmax < 0.2, the reactive power should not exceed
            10%/20% of the maximum active power.
        Args:
            cos_phi (float): The constant power factor value between 0.90 and 1.
            mom_p_mw -- momentary active power desired by run_battery_p_control
        Returns:
            self.p_mw -- active power calculated by reactive power control for DER-Inverter
            self.q_mvar -- reactive power calculated by reactive power control for DER-Inverter
        """
        # Normalize momentary active power output to its max value
        self.norm_mom_p_mw = mom_p_mw / self.max_p_mw
        # Ensure normalized momentary active power is not greater than the max
        self.norm_mom_p_mw = min(self.norm_mom_p_mw, self.norm_max_p_mw)
        # Set momentary cos phi to the minimum of min cos phi and nominal cos phi
        self.mom_cos_phi = min(self.min_cos_phi, self.nomi_cos_phi)
        # Check if cos phi is within the range [0.90, 1] and raise error if not
        if self.mom_cos_phi < self.min_cos_phi or self.mom_cos_phi > 1:
            raise ValueError("cos_phi must be in the range [0.90, 1]")
        # Set normalized momentary apparent power to the minimum of normalized values
        self.norm_mom_sn_mva = min(self.norm_mom_p_mw, self.norm_sn_mva)
        # Calculate active and reactive power from momentary normalized apparent power and cos phi
        self.norm_mom_p_mw, q_norm_from_cos_phi = self._pq_from_cos_phi(
            s_mva=self.norm_mom_sn_mva,
            cos_phi=self.mom_cos_phi,
            qmode=self.qmode,
            pmode=self.pmode,
        )
        # Ensure normalized reactive power is within inverter range
        self.norm_q_mvar = max(
            self.norm_min_q_mvar, min(q_norm_from_cos_phi, self.norm_max_q_mvar)
        )
        # Calculate normalized active power from apparent and reactive power
        self.norm_p_mw = min(
            self.norm_mom_p_mw, np.sqrt(self.norm_sn_mva**2 - self.norm_q_mvar**2)
        )
        # Calculate momentary apparent power from active and reactive power
        self.mom_sn_mva = np.sqrt(self.norm_p_mw**2 + self.norm_q_mvar**2)
        # Check Low Power Range Condition
        if np.abs(self.norm_mom_p_mw) <= self.reactive_power_performance_limit:
            # Store the unbounded reactive power value
            self.raw_norm_q_mvar = self.norm_q_mvar

            # Regulate reactive power to be no more than 10% of max active power, as in VDE.
            max_bounded_q_mvar = (
                self.norm_p_mw * self.ten_prct_reactive_power_performance_limit
            )
            q_mvar_abs = min(abs(self.norm_q_mvar), abs(max_bounded_q_mvar))

            # Update the reactive power and normalize respecting the limitation.
            self.q_mvar = np.copysign(q_mvar_abs, self.norm_q_mvar) * self.sn_mva
            self.norm_q_mvar = self.q_mvar / self.sn_mva

            # Update real power with the condition that it should not exceed the square root
            # of the difference of the square of apparent power and the square of reactive power.
            self.p_mw = min(mom_p_mw, np.sqrt(self.sn_mva**2 - self.q_mvar**2))
        else:
            # Active power exceeds 10% or 20% of the maximum active power.
            # Fixed cos_phi(P) curve is based on a fixed cos_phi value.
            # It follows a piecewise linear function for active power.
            self.p_mw = self.norm_p_mw * self.sn_mva
            self.q_mvar = self.norm_q_mvar * self.sn_mva

    def _calculate_q_out(self):
        """Calculates reactive power output based on measured voltage.
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
        """Calculates active and reactive power from power factor."""
        # Get the signs for reactive and active power modes
        p_sign = self.get_p_sign(pmode)
        q_sign = self.get_q_sign(qmode)
        # Calculate active power
        p_mw = s_mva * cos_phi
        # Calculate reactive power
        q_mvar = p_sign * q_sign * np.sqrt(s_mva**2 - p_mw**2)
        return p_mw, q_mvar

    def rbc_pvbes_decentralized_sc_ctrl(self, net, p_mw=0):
        """PV-BES Strategy 1 -- Decentralized SC:
        - Implements a Rule-Based Control (RBC) algorithm for home batteries to
          support solar power integration into low-voltage grids.
        - Enhances self-consumption by storing excess PV energy generated during sunny hours.
        - Economically important when feed-in tariff for PV electricity is lower
          than grid purchase price.
        - Primary role of PV: Covers local load; surplus energy stored in Battery
          Energy Storage System (BES).
        - Residual power: Difference between PV generation and load demand at PCC.
        - Grid Feed-In: When PV generation is present and SoC^max limit is reached,
          supplies excess power to the grid, aiding in voltage control.
        - Battery's role: Ensures self-sufficiency at PCC, regulates only active
          power for self-consumption.
        INPUT:
           **net** (pandapowerNet)
           **time** (int) - Current time step of the simulation
        OUTPUT:
           **p_mw** (float) - Returns Charging/Discharging/Idle power of the
           battery in MW
        """
        # Selects all local sgens in the same bus as the battery
        local_mask_sgen = net.sgen.bus == self.bus
        # Selects all local loads in the same bus as the battery
        local_mask_load = net.load.bus == self.bus
        # Calculates the total power generation at the same bus
        pv_power = net.sgen.p_mw[local_mask_sgen].sum()
        # Calculates the total power consumption at the same bus
        load_power = net.load.p_mw[local_mask_load].sum()
        residual_power = pv_power - load_power
        # (I) Battery Charging Block:
        if (self.soc_percent < self.soc_percent_charge_limit) and (residual_power > 0):
            # positive charging
            p_charge_mw = self.max_p_mw if pv_power > self.max_p_mw else pv_power
            p_charge_mw = min(self.max_p_mw, pv_power)
            p_mw = p_charge_mw
        # (II) Battery Discharging Block:
        elif (self.soc_percent > self.soc_percent_discharge_limit) and (
            residual_power < 0
        ):
            # negative discharging
            p_discharge_mw = (
                self.min_p_mw if load_power < self.min_p_mw else -load_power
            )
            p_discharge_mw = max(self.min_p_mw, -load_power)
            p_mw = p_discharge_mw
        else:
            p_mw = 0
        net.storage.at[self.sid, "p_mw"] = p_mw
        return p_mw

    def rbc_pvbes_distributed_sc_ctrl(self, net, p_mw=0):
        """PV-BES Strategy 2 -- Distributed SC:
        - Implements a distributed PV-BES strategy in which the LV grid is
          divided into control zones.
        - Each operator manages a control zone, extending from the transformer
          substation to the end of the feeder.
        - For each time step 't', calculates the total power consumption and PV
          generation within the zone to determine the residual power.
        - If there's a power surplus and the SoCs of the BESs are below their
          maximum levels, the BESs charge at a rate dependent on the residual
          power and the maximum charging rate.
        - When SoCs reach their maximum levels, PVs supply only surplus power
          to the main grid.
        - Conversely, with a power deficit and the BESs above their minimum
          levels, the BESs discharge based on the residual power and the
          maximum discharging rate, sourcing any shortfall from the grid.
        INPUT:
           **net** (pandapowerNet) - Contains the pandapower net of a power system
        OUTPUT:
            **p_mw** (float) - Returns Charging/Discharging/Idle power of the
            battery in MW.
        """
        # Create masks for sgen, load and storage components based on the zone
        mask_sgen = net.sgen.zone == self.zone
        mask_load = net.load.zone == self.zone
        # Calc the sum of PV,load and storage generation in the zone
        sum_pv_zone_p_mw = net.sgen.p_mw[mask_sgen].sum()
        sum_load_zone_p_mw = net.load.p_mw[mask_load].sum()
        # sum_storage_zone_p_mw = net.storage.p_mw[mask_storage].sum()
        # Calc difference of PV-Power and Load-Consum within Zone
        residual_power = sum_pv_zone_p_mw - sum_load_zone_p_mw
        # (I) Battery Charging Block:
        if (self.soc_percent < self.soc_percent_charge_limit) and residual_power > 0:
            # positive charging
            p_charge_mw = (
                self.max_p_mw if residual_power > self.max_p_mw else residual_power
            )
            p_mw = p_charge_mw
        # (II) Battery Discharging Block:
        elif (
            self.soc_percent > self.soc_percent_discharge_limit
            and residual_power <= 0.0
        ):
            # negative discharging
            p_discharge_mw = (
                self.min_p_mw if residual_power < self.min_p_mw else residual_power
            )
            p_mw = p_discharge_mw
        else:
            p_mw = 0
        net.storage.at[self.sid, "p_mw"] = p_mw
        return p_mw

    def rbc_pvbes_distributed_sc_dnc_ctrl(self, net, time, p_mw=0):
        """PV-BES Strategy 3 -- Distributed SC+DNC:
        - Uses the same functionality as the 'PV-BES Strategy 2' but time-dependent.
        - Combines Demand Network Control (DNC) and Self-Consumption (SC)
          methodologies.
        - During configurable daylight hours, typically from 6:00 a.m. to
          6:00 p.m., stores excess PV power to optimize self-consumption; power
          discharges at night.
        - Ensures time-aligned charging/discharging cycles for maximized
          self-consumption by leveraging optimal PV generation.
        - Specific charging and discharging operations are based on the Charge
          and Discharge Blocks from Algorithm 'PV-BES Strategy 2'.
        INPUT:
           **net** (pandapowerNet) - Contains the pandapower net of a power system
           **time** (int) - The current time step of the simulation.
        OUTPUT:
            **p_mw** (float) - Returns Charging/Discharging/Idle power of the
            battery in MW.
        """
        # Create masks for sgen, load and storage components based on the zone
        mask_sgen = net.sgen.zone == self.zone
        mask_load = net.load.zone == self.zone
        mask_storage = net.storage.zone == self.zone
        # Calc the sum of PV,load and storage generation in the zone
        sum_pv_zone_p_mw = net.sgen.p_mw[mask_sgen].sum()
        sum_load_zone_p_mw = net.load.p_mw[mask_load].sum()
        sum_storage_zone_p_mw = net.storage.p_mw[mask_storage].sum()
        # Calc difference of PV-Power and Load-Consum
        residual_power = sum_pv_zone_p_mw - sum_load_zone_p_mw + sum_storage_zone_p_mw
        # Calc time-based control scheduling
        time_minute = (time - 1) * 15
        hour = time_minute // 60
        flag_time = (0 <= hour % 24 < 8) or (18 <= hour % 24 < 24)
        # (I) Battery Charging Block:
        if (self.soc_percent < self.soc_percent_charge_limit) and (residual_power > 0):
            # positive charging
            p_charge_mw = (
                self.max_p_mw if residual_power > self.max_p_mw else residual_power
            )
            p_mw = p_charge_mw
        # (II) Battery Discharging Block:
        elif (
            self.soc_percent > self.soc_percent_discharge_limit
            and residual_power <= 0
            and flag_time
        ):
            # negative discharging
            p_discharge_mw = (
                self.min_p_mw if residual_power < self.min_p_mw else residual_power
            )
            p_mw = p_discharge_mw
        else:
            p_mw = 0
        net.storage.at[self.sid, "p_mw"] = p_mw
        return p_mw

    def rbc_bes_dnc_strategy_ctrl(self, net, time, p_mw=0):
        """BES Strategy 1 -- DNC:
        - Operates based on preset charging and discharging cycles depending on
          the time of day.
        - Charges from 6:00 a.m. to 6:00 p.m., if the SoC hasn't reached its
          maximum limit (SoC^max).
        - Discharges from 6:00 p.m. to 6:00 a.m. next day, as long as the SoC
          is above its minimum threshold (SoC^min).
        - Configurable timing enhances operational flexibility.
        - p_t^{cha-max} and p_t^{dis-max} denote the charging and discharging
          power of the BES system, respectively.
        INPUT:
           **net** (pandapowerNet) - Contains the pandapower net of a power system
           **time** (int) - The current time step of the simulation.
        OUTPUT:
           **p_mw** (float) - Returns Charging/Discharging/Idle power of the battery in MW.
        """
        # Calculate the current hour of the day
        time_minute = (time - 1) * 15  # Assuming each time step is 15 minutes
        hour = time_minute // 60

        # Determine charging or discharging phase
        is_charging_time = 6 <= hour % 24 < 18  # Charging from 6:00 a.m. to 6:00 p.m.
        is_discharging_time = (
            not is_charging_time
        )  # Discharging from 6:00 p.m. to 6:00 a.m.

        # (I) Battery Charging Block:
        if is_charging_time and self.soc_percent < self.soc_percent_charge_limit:
            # positive charging
            p_charge_mw = min(
                self.max_p_mw, net.sgen.p_mw.sum()
            )  # Charge with available solar power or max capacity
            p_mw = p_charge_mw
        # (II) Battery Discharging Block:
        elif (
            is_discharging_time and self.soc_percent > self.soc_percent_discharge_limit
        ):
            # negative discharging
            p_discharge_mw = max(
                self.min_p_mw, -net.load.p_mw.sum()
            )  # Discharge to meet load or min capacity
            p_mw = p_discharge_mw
        else:
            # Idle state
            p_mw = 0
        # Update power output of the storage
        net.storage.at[self.sid, "p_mw"] = p_mw
        return p_mw
