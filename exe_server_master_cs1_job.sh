#!/bin/bash
# Script Name: exe_server_master_cs1_job.sh
# Author: Demirel
# Description: Orchestrates and submits batch jobs in an HPC environment with unique JOB_IDs (1-5), customized via 'exe_server_helper_cs1_job.sh'. 
# Manages resources, time limits, and notifications for efficient execution.
# -----------------------------------------
# Usage Comments
# chmod +x ./exe_server_master_cs1_job.sh 
# Use this command to grant execute permissions to the script. This ensures the script can be executed directly from the command line.

# ./exe_server_master_cs1_job.sh
# Execute the script locally without submitting it to the SLURM scheduler. Useful for testing or debugging the script in a non-HPC environment.

# sbatch ./exe_server_master_cs1_job.sh 
# Submit the script as a SLURM batch job in a High-Performance Computing (HPC) environment. The SLURM scheduler will manage the execution of the script on available compute nodes.
# -----------------------------------------
# Define base directory for execution and output directories for logs
BASE_HELPER_DIR="/hkfs/home/haicore/iai/ii6824/Documents/PIDE/exe_server_helper_cs1_job.sh" # Define the directory for helper job execution
OUTPUT_DIR="output/jobs" # Define the output directory for job logs 
LOG_FILE="output/jobs/exe_server_master_cs1_job.log" # Define the log file to record script execution

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Redirect output to log file and console
{
    echo "Master script started at $(date)" # Start message
    # Function to submit jobs with custom parameters
    submit_job_for_group() {
        local JOB_ID="$1"
        echo "GROUP_ID: ${JOB_ID}"
        sbatch --nodes=1 \
               --partition=normal \
               --time=70:00:00 \
               --gres=gpu:2g.10gb:1 \
               --mail-user=goekhan.demirel@kit.edu \
               --mail-type=BEGIN,END,TIME_LIMIT_80 \
               --output=output/jobs/slurm-${JOB_ID}-%j.out \
               --error=output/jobs/slurm-${JOB_ID}-%j.err \
               --job-name=PIDE-run-job-${JOB_ID} \
               $BASE_HELPER_DIR "$@"
    }
    # Job submissions with JOB_IDs 1 to 5:
    # JOB_ID 1 - 0
    submit_job_for_group 1 \
                         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
                         600 600 600 600 800 800 800 800 1000 1000 1000 1000 \
                         800 1200 1600 2000 800 1200 1600 2000 800 1200 1600 2000
    
    # JOB_ID 2 - 25
    submit_job_for_group 2 \
                         25.0 25.0 25.0 25.0 25.0 25.0 25.0 25.0 25.0 25.0 25.0 25.0 \
                         600 600 600 600 800 800 800 800 1000 1000 1000 1000 \
                         800 1200 1600 2000 800 1200 1600 2000 800 1200 1600 2000
    # JOB_ID 3 - 50
    submit_job_for_group 3 \
                         50.0 50.0 50.0 50.0 50.0 50.0 50.0 50.0 50.0 50.0 50.0 50.0 \
                         600 600 600 600 800 800 800 800 1000 1000 1000 1000 \
                         800 1200 1600 2000 800 1200 1600 2000 800 1200 1600 2000
    
    # JOB_ID 4 - 75
    submit_job_for_group 4 \
                         75.0 75.0 75.0 75.0 75.0 75.0 75.0 75.0 75.0 75.0 75.0 75.0 \
                         600 600 600 600 800 800 800 800 1000 1000 1000 1000 \
                         800 1200 1600 2000 800 1200 1600 2000 800 1200 1600 2000
    # JOB_ID 5 - 5
    submit_job_for_group 5 \
                         100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 \
                         600 600 600 600 800 800 800 800 1000 1000 1000 1000 \
                         800 1200 1600 2000 800 1200 1600 2000 800 1200 1600 2000
    
    echo "Master script finished at $(date)" # End message
} | tee "$LOG_FILE"