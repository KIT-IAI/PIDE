"""Main module for PIDE: Photovoltaic Integration Dynamics and Efficiency.
This script provides the configuration and execution flow for simulating and evaluating power grid scenarios.
"""

import sys
import os
import argparse
import distutils.util
import pickle
import logging
from typing import Optional, List

# Import helper classes for power grid control and evaluation
from helper.helper_powergrid_control_rbc_pp import PowerGridRuleBasedControlPP
from helper.helper_powergrid_evaluator import PowerGridEvaluator

# Define command-line argument parser
parser = argparse.ArgumentParser(
    description="PIDE: Photovoltaic Integration Dynamics and Efficiency for Autonomous Control on Power Distribution Grids."
)

# Benchmark and scenario configuration
parser.add_argument(
    "--benchmark",
    type=str,
    default="simbench",
    help="Benchmark dataset: 'simbench', 'customised'.",
)
parser.add_argument(
    "--sb_code",
    type=str,
    nargs="?",
    default="1-LV-rural1--0-sw",
    help="Simbench code for the benchmark dataset.",
)
parser.add_argument(
    "--scenario",
    type=str,
    nargs="?",
    default="2-future",
    help="Scenario options: '0-today', '1-near future', '2-future'.",
)

# Standards and regulation modes
parser.add_argument(
    "--standard",
    type=str,
    default="vde",
    help="Standards: 'vde' (VDE-AR-N 4105) or 'customised' (e.g., IEEE Std 1547-2018).",
)
parser.add_argument(
    "--standard_mode",
    type=str,
    default="base",
    help="Mode options: 'base', 'deadband', or 'customised'.",
)

# Control modes for timeseries, PV, and storage
parser.add_argument(
    "--timeseries_ctrl",
    type=str,
    default="control_module",
    help="Timeseries control module.",
)
parser.add_argument(
    "--pv_ctrl",
    type=str,
    default="voltage_reactive_power_ctrl",
    help="Control mode for PV systems.",
)
parser.add_argument(
    "--storage_p_ctrl",
    type=str,
    default="rbc_pvbes_distributed_sc_ctrl",
    help="P-Control mode for storage systems.",
)
parser.add_argument(
    "--storage_q_ctrl",
    type=str,
    default="voltage_reactive_power_ctrl",
    help="Q-Control mode for storage systems.",
)
parser.add_argument(
    "--soc_initial",
    type=float,
    default=0.0,
    help="Initial state of charge (SoC) for storage systems.",
)

# Scaling parameters for PV, load, and storage
parser.add_argument(
    "--scaling_pv",
    type=float,
    default=1.0,
    help="Scaling factor for PV capacity (0.0-10.0).",
)
parser.add_argument(
    "--scaling_load",
    type=float,
    default=1.0,
    help="Scaling factor for load capacity (0.0-10.0).",
)
parser.add_argument(
    "--scaling_storage",
    type=float,
    default=1.0,
    help="Scaling factor for storage capacity (0.0-10.0).",
)

# Simulation time settings
parser.add_argument(
    "--time_mode",
    type=str,
    default="selected",
    help="Time mode: 'selected', 'random', or 'default'.",
)
parser.add_argument(
    "--episode_start_hour",
    type=int,
    choices=range(0, 25),
    default=0,
    help="Start hour (0-24).",
)
parser.add_argument(
    "--episode_start_day",
    type=int,
    choices=range(0, 355),
    default=180,
    help="Start day (0-354).",
)
parser.add_argument(
    "--episode_start_min_interval",
    type=int,
    choices=range(0, 4),
    default=0,
    help="Starting interval (0-3).",
)
parser.add_argument(
    "--episode_limit",
    type=int,
    choices=range(0, 35136),
    default=96,
    help="Maximum number of time steps in an episode.",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    choices=range(0, 35136),
    default=35135,
    help="Maximum number of iterations for the simulation.",
)

# Monte Carlo simulation settings
parser.add_argument(
    "--flag_monte_carlo",
    type=lambda x: bool(distutils.util.strtobool(x)),
    default="true",
    help="Enable or disable Monte Carlo simulation mode.",
)
parser.add_argument(
    "--num_monte_carlo_runs",
    type=int,
    default=2,  # 50
    help="Number of Monte Carlo simulation runs.",
)
parser.add_argument(
    "--seed_value",
    type=int,
    choices=range(0, 100),
    default=42,
    help="Initial seed value for Monte Carlo simulations (range 0-100).",
)

# Mini Photovoltaic (MPV) settings
parser.add_argument(
    "--mpv_flag",
    type=lambda x: bool(distutils.util.strtobool(x)),
    default="true",
    help="Enable or disable MPV analysis.",
)
parser.add_argument(
    "--mpv_benchmark",
    type=str,
    default="simbench",
    help="MPV benchmarking source: 'simbench', 'mpvbench', or 'customised'.",
)
parser.add_argument(
    "--mpv_scaling",
    type=float,
    default=0.60,
    help="Scaling factor for MPV capacity (0.0-10.0).",
)
parser.add_argument(
    "--mpv_concentration_rate_percent",
    type=float,
    default=100.00,
    help="MPV concentration rate as a percentage (0.0-100.0).",
)
parser.add_argument(
    "--mpv_inverter_apparent_power_watt",
    type=int,
    choices=range(600, 1001),
    default=800,
    help="Maximum apparent power for MPV inverters (600-1000 W).",
)
parser.add_argument(
    "--mpv_solar_cell_capacity_watt",
    type=int,
    choices=range(800, 2001),
    default=2000,
    help="Maximum power for MPV solar cells (800-2000 W).",
)

# Configuration file settings
parser.add_argument(
    "--cfg_default",
    type=str,
    default="default_config_grid_manage_der",
    help="Default configuration file.",
)
parser.add_argument(
    "--cfg_user",
    type=str,
    default="user_config_grid_manage_der",
    help="User-defined configuration file.",
)

# Plot configuration
parser.add_argument(
    "--output_path",
    type=str,
    default="local",
    help="Output path for evaluation results.",
)
parser.add_argument(
    "--plattform",
    type=str,
    default="hpc",
    choices=["haicore", "local"],
    help="Platform type: 'haicore' or 'local'.",
)
parser.add_argument(
    "--cfg_plot_default",
    type=str,
    default="default_plot_config",
    help="Default plot configuration file.",
)
parser.add_argument(
    "--cfg_plot_user",
    type=str,
    default="user_plot_config_grid",
    help="User-defined plot configuration file.",
)

# Parse command-line arguments
arguments = parser.parse_args()

# Configure paths based on the current directory
base_path = os.getcwd()
arguments.base_path = os.path.join(base_path)
arguments.output_path = os.path.join(base_path, "output", arguments.output_path)
arguments.yaml_path = os.path.join(base_path, "yaml", arguments.plattform)

# Paths for configuration files
arguments.cfg_default_path = os.path.join(
    arguments.yaml_path, arguments.cfg_default + ".yaml"
)
arguments.cfg_user_path = os.path.join(
    arguments.yaml_path, arguments.cfg_user + ".yaml"
)

# Paths for plot configuration files
arguments.cfg_default_plot_path = os.path.join(
    arguments.yaml_path, arguments.cfg_plot_default + ".yaml"
)
arguments.cfg_user_plot_path = os.path.join(
    arguments.yaml_path, arguments.cfg_plot_user + ".yaml"
)

# Path for sequence ID file
arguments.sequence_id_path = os.path.join(arguments.yaml_path, "sequence_id.txt")


def print_argument_settings(arguments):
    """Prints the settings of command-line arguments."""
    print("--------------- Argument Settings ---------------")
    print(f"{'Setting':<35}{'Value'}")
    print(f"{'-'*50}")
    print(f"{'Episode Limit:':<35}{arguments.episode_limit}")
    print(f"{'Flag Monte Carlo:':<35}{arguments.flag_monte_carlo}")
    print(f"{'Num Monte Carlo Samples:':<35}{arguments.num_monte_carlo_runs}")
    print(f"{'MPV Flag:':<35}{arguments.mpv_flag}")
    print(
        f"{'MPV Concentration Rate (%):':<35}{arguments.mpv_concentration_rate_percent}"
    )
    print(
        f"{'MPV Inverter Apparent Power (Watt):':<35}{arguments.mpv_inverter_apparent_power_watt}"
    )
    print(
        f"{'MPV Solar Cell Capacity (Watt):':<35}{arguments.mpv_solar_cell_capacity_watt}"
    )


def main_sim(args):
    """Main simulation function."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Parsing Arguments")
    print_argument_settings(arguments)
    logging.info("Executing PowerGridRuleBasedControlPP")

    # Create folder for simulation results
    folder_name = "output/mcs_folder"
    os.makedirs(folder_name, exist_ok=True)

    raw_log_mcs_data = []
    if arguments.flag_monte_carlo:
        # Monte Carlo simulations
        print("------------------ Monte Carlo Simulation ------------------\n")
        for i in range(arguments.num_monte_carlo_runs):
            arguments.seed_value += i
            print(f" Monte Carlo Run {i+1} of {arguments.num_monte_carlo_runs}")
            print(f" Current Seed Value: {arguments.seed_value}")
            pg_rbc_pp = PowerGridRuleBasedControlPP(arguments)
            logging.info("Running Monte Carlo simulation")
            try:
                pg_rbc_pp.run_main()
            except Exception as e:
                logging.error(f"Error during simulation: {e}")
                continue
            # Save simulation data
            raw_log_mcs_data.append(
                {"simulation": pg_rbc_pp.output, "arguments": arguments}
            )
            file_path = os.path.join(
                folder_name, f"{pg_rbc_pp.extended_path}_monte_carlo_sim_{i+1}.pkl"
            )
            with open(file_path, "wb") as f:
                pickle.dump(raw_log_mcs_data[-1], f)
            print(f"Saved Monte Carlo run {i+1} to: {file_path}")
        print("------------------ Monte Carlo Simulation Completed ------------------")
    else:
        # Standard simulation
        pg_rbc_pp = PowerGridRuleBasedControlPP(arguments)
        logging.info("Running standard simulation")
        pg_rbc_pp.run_main()
        print("------------------ Standard Simulation Completed ------------------")

    pg_rbc_pp.raw_log_mcs_data = raw_log_mcs_data
    return pg_rbc_pp


def main_eval(pg_rbc_pp):
    """Evaluation and visualization."""
    if arguments.timeseries_ctrl != "test_mode":
        logging.info("Running PowerGridEvaluator")
        pg_rbc_pp_pge = PowerGridEvaluator(pg_rbc_pp)
    else:
        logging.info("Test mode active, skipping evaluation")
        pg_rbc_pp_pge = None
    return pg_rbc_pp, pg_rbc_pp_pge


if __name__ == "__main__":
    pg_rbc_ppa = main_sim(arguments)
