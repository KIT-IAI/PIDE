#!/bin/bash
# Script Name: exe_server_master_cs3_job.sh
# Author: Demirel
# Description: Advanced script for running VDE control case study simulations with additional features including dynamic control strategy selection, dry run option, and custom input handling.

#SBATCH --nodes=1 
#SBATCH --partition=normal
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:full:1 
#SBATCH --mem=501600mb
#SBATCH --mail-user=goekhan.demirel@kit.edu
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_80
#SBATCH --output=output/jobs/slurm-exe_server_master_cs3_job-%j.out
#SBATCH --error=output/jobs/slurm-exe_server_master_cs3_job-%j.err
#SBATCH --job-name=PIDE-cs3-job
# -----------------------------------------
# Usage Comments
# chmod +x ./exe_server_master_cs3_job.sh
# Use this command to grant execute permissions to the script. This ensures the script can be executed directly from the command line.

# ./exe_server_master_cs3_job.sh
# Execute the script locally without submitting it to the SLURM scheduler. Useful for testing or debugging the script in a non-HPC environment.

# sbatch ./exe_server_master_cs3_job.sh
# Submit the script as a SLURM batch job in a High-Performance Computing (HPC) environment. The SLURM scheduler will manage the execution of the script on available compute nodes.
# -----------------------------------------

# Logging setup and Default configurations
LOG_FILE="output/mcs_folder/exe_server_master_cs3_job.log"
OUTPUT_NB_DIR="/hkfs/home/haicore/iai/ii6824/PIDE/output/mcs_folder"
PYTHON_ENV="../pide_env"

echo "Log File for PIDE Case Study 3 Script" > "$LOG_FILE"  # Reset log file at start
exec 3>&1 1>>"$LOG_FILE" 2>&1  # Redirect stdout and stderr to log file and fd 3

# Function to log messages with timestamps
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >&3
}

# log_message "Starting eGrid VDE Control Case Study 2 and 3 Script"
log_message "Starting PIDE Case Study 3 Script"


DRY_RUN=false

# Parse command line options
while getopts "e:o:d" opt; do
  case $opt in
    e) PYTHON_ENV="$OPTARG"
    ;;
    o) OUTPUT_NB_DIR="$OPTARG"
    ;;
    d) DRY_RUN=true
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Check Python environment
if [ ! -d "$PYTHON_ENV" ]; then
    log_message "Error: Python environment at $PYTHON_ENV not found!"
    exit 1
fi

# Variable initialization
soc_initial=0.0
episode_start_day=180
episode_limit=96 
flag_monte_carlo=true # Set to False if Monte Carlo simulations are not required
num_monte_carlo_runs=100 # Number of Monte Carlo simulation samples (default: 100)
mpv_flag=true
mpv_concentration_rate_percent=100.00
mpv_inverter_apparent_power_watt=800
mpv_solar_cell_capacity_watt=2000
output_path="$OUTPUT_NB_DIR"

# Dynamically selecting control strategies based on user input or default
#-------------------------------------------------------------------------
# Part 3:
pv_ctrl_list=("voltage_reactive_power_ctrl" )
storage_p_ctrl_list=("rbc_pvbes_decentralized_sc_ctrl" "rbc_pvbes_distributed_sc_ctrl" "rbc_pvbes_distributed_sc_dnc_ctrl" "rbc_bes_dnc_ctrl")
    # (1): ''datasource' RawData
    # (2): 'rbc_pvbes_decentralized_sc_ctrl'
    # (3): 'rbc_pvbes_distributed_sc_ctrl'
    # (4): 'rbc_pvbes_distributed_sc_dnc_ctrl'
    # (5): 'rbc_bes_dnc_ctrl'
# Activate Python environment
log_message "Activating Python environment at $PYTHON_ENV"
source "$PYTHON_ENV/bin/activate"
# Initialize ID counter
id=0
# Main simulation loop
for pv_ctrl in "${pv_ctrl_list[@]}"; do
    for storage_p_ctrl in "${storage_p_ctrl_list[@]}"; do
        ((id++))
        log_message "Preparing simulation ID: $id with pv_ctrl = $pv_ctrl, storage_p_ctrl = $storage_p_ctrl"
        if [ "$DRY_RUN" = true ]; then
            log_message "Dry run enabled. Skipping actual execution for simulation ID: $id."
            continue
        fi
        # Execute the simulation with specified parameters
        log_message "Executing simulation ID: $id"
        cmd="python __main__.py --soc_initial $soc_initial --episode_start_day $episode_start_day --episode_limit $episode_limit --mpv_flag $mpv_flag --flag_monte_carlo $flag_monte_carlo --num_monte_carlo_runs $num_monte_carlo_runs --mpv_concentration_rate_percent $mpv_concentration_rate_percent --mpv_inverter_apparent_power_watt $mpv_inverter_apparent_power_watt --mpv_solar_cell_capacity_watt $mpv_solar_cell_capacity_watt --output_path $output_path --pv_ctrl $pv_ctrl --storage_p_ctrl $storage_p_ctrl"
        log_message "CMD: $cmd"
        eval $cmd
        if [ $? -eq 0 ]; then
            log_message "Simulation ID: $id completed successfully."
        else
            log_message "Error occurred during simulation ID: $id."
            exit 1
        fi
    done
done
log_message "All simulations for Case Study 3 have been successfully completed."
