#!/bin/bash
# Script Name: exe_server_helper_cs1_job.sh
# Author: Demirel
# Description: Executes Jupyter notebooks for sensitivity analysis in an HPC environment, triggered by the master script. Automates and streamlines analysis workflows efficiently.
# -----------------------------------------
# Usage Comments
# chmod +x ./exe_server_helper_cs1_job.sh
# Use this command to grant execute permissions to the script. This ensures the script can be executed directly from the command line.

# ./exe_server_helper_cs1_job.sh
# Execute the script locally without submitting it to the SLURM scheduler. Useful for testing or debugging the script in a non-HPC environment.

# sbatch ./exe_server_helper_cs1_job.sh 
# Submit the script as a SLURM batch job in a High-Performance Computing (HPC) environment. The SLURM scheduler will manage the execution of the script on available compute nodes.
# -----------------------------------------
echo "------------------------------"
echo "Script Start"
echo "------------------------------"

# Define output directory and environment paths
BASE_DIR="/hkfs/home/haicore/iai/ii6824/PIDE"
OUTPUT_NB_DIR="/hkfs/home/haicore/iai/ii6824/PIDE/output/notebook"
OUTPUT_DIR="/hkfs/home/haicore/iai/ii6824/PIDE/output"
PYTHON_ENV="../pide_env"
JUPYTER_NOTEBOOK="/hkfs/home/haicore/iai/ii6824/PIDE/execution_script_cs1.ipynb" # Execution script over one year  execution_script_cs1

# Logging setup
# Generate a log file name with current date and hour
CURRENT_DATE_TIME=$(date "+%Y-%m-%d_%H-%M")
LOG_FILE="/hkfs/home/haicore/iai/ii6824/Documents/PIDE/output/notebook/log_file-exe_server_helper_cs1_job_$CURRENT_DATE_TIME.log"
echo "Log File for exe_server_helper_cs1_job.sh, $CURRENT_DATE_TIME" > "$LOG_FILE"  # Reset log file at start
exec 3>&1 1>>"$LOG_FILE" 2>&1  # Redirect stdout and stderr to log file and fd 3

# Function to log messages with timestamps
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >&3
}

log_message "Starting Script"

# Check if Python environment exists
if [ ! -d "$PYTHON_ENV" ]; then
    log_message "Error: Python environment $PYTHON_ENV not found!"
    exit 1
else
    log_message "Python environment $PYTHON_ENV found."
fi

if [ ! -f "$JUPYTER_NOTEBOOK" ]; then
    log_message "Error: Jupyter Notebook $JUPYTER_NOTEBOOK not found!"
    exit 1
else
    log_message "Jupyter Notebook $JUPYTER_NOTEBOOK found."
fi

# Create output directory if it does not exist
if [ ! -d "$OUTPUT_NB_DIR" ]; then
    mkdir -p $OUTPUT_NB_DIR
    log_message "Created output directory $OUTPUT_NB_DIR."
else
    log_message "Output directory $OUTPUT_NB_DIR already exists."
fi

# Activate Python environment
log_message "Activating Python environment $PYTHON_ENV."
source $PYTHON_ENV/bin/activate
if [ $? -eq 0 ]; then
    log_message "Python environment $PYTHON_ENV activated successfully."
else
    log_message "Failed to activate Python environment $PYTHON_ENV."
    exit 1
fi

# Before executing each notebook
log_message "Executing Jupyter Notebook with parameters: MPV_CONCENTRATION_RATE_PERCENT=$CONCENTRATION, MPV_INVERTER_APPARENT_POWER_WATT=$INVERTER_POWER, MPV_SOLAR_CELL_CAPACITY_WATT=$SOLAR_CAPACITY, ID=$ID."

# Counter variable for the ID
ID=0
# Read arguments/parameters from the command line
GROUP_ID="$1"
CONCENTRATION_VALUES=("${@:2:13}")
INVERTER_POWER_VALUES=("${@:14:13}")
SOLAR_CAPACITY_VALUES=("${@:26:13}")

echo "------------------------------"
echo "GROUP_ID $GROUP_ID"
echo "------------------------------"
echo "OUTPUT_NB_DIR $OUTPUT_NB_DIR"
echo "JUPYTER_NOTEBOOK $JUPYTER_NOTEBOOK"
echo "------------------------------"

# Main part of the script
for i in {0..11}; do
    CONCENTRATION=${CONCENTRATION_VALUES[i]}
    INVERTER_POWER=${INVERTER_POWER_VALUES[i]}
    SOLAR_CAPACITY=${SOLAR_CAPACITY_VALUES[i]}
    # Convert the concentration value to an integer for the file name
    CONCENTRATION_INT=$(printf "%.0f" "$CONCENTRATION")

    echo "GROUP_ID:$GROUP_ID LOOP:$ID"
    # Convert the concentration value to an integer for the file name
    CONCENTRATION_INT=$(printf "%.0f" "$CONCENTRATION")
    # Execute the Jupyter Notebook with different arguments
    echo "Executing Jupyter Notebook with MPV_CONCENTRATION_RATE_PERCENT=$CONCENTRATION, MPV_INVERTER_APPARENT_POWER_WATT=$INVERTER_POWER, MPV_SOLAR_CELL_CAPACITY_WATT=$SOLAR_CAPACITY, ID=$ID, OUTPUT_NB_DIR=$OUTPUT_NB_DIR"
    OUTPUT_FILE="$OUTPUT_NB_DIR/run_nb_${GROUP_ID}_ID${ID}_${CONCENTRATION_INT}_${INVERTER_POWER}_${SOLAR_CAPACITY}.ipynb"
    # Inside the for loop where you call papermill
    papermill "$JUPYTER_NOTEBOOK" "$OUTPUT_FILE" \
        -p MPV_CONCENTRATION_RATE_PERCENT "$CONCENTRATION" \
        -p MPV_INVERTER_APPARENT_POWER_WATT "$INVERTER_POWER" \
        -p MPV_SOLAR_CELL_CAPACITY_WATT "$SOLAR_CAPACITY" \
        -p ID "$ID" \
        -p OUTPUT_NB_DIR "$OUTPUT_NB_DIR"

    # After executing each notebook
    if [ $? -eq 0 ]; then
        log_message "Successfully executed notebook $OUTPUT_FILE."
    else
        log_message "Failed to execute notebook $OUTPUT_FILE."
        # Consider adding a command to handle the failure, like exiting or continuing with the next iteration
    fi

    # Incrementing the ID
    ((ID++))

done

# Execute the JUPYTER_NOTEBOOK_REFINER at the end of the script
echo "------------------------------"
echo "Executing the Jupyter Notebook Refiner"
echo "------------------------------"
OUTPUT_REFINER_FILE="$OUTPUT_DIR/01a_NB_for_CaseStudy_1_Refiner.ipynb" # BASE_DIR
papermill "$JUPYTER_NOTEBOOK_REFINER" "$OUTPUT_REFINER_FILE"

# End of the script
echo "------------------------------"
echo "Script Finished"
echo "------------------------------"