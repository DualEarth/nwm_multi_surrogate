#!/bin/bash
#SBATCH --job-name=nwm_range_download # Name of your job
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --partition=main            # Specify the partition
#SBATCH --qos=main                  # Specify the Quality of Service (QoS)
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=32          # Number of CPU cores per task (adjust max_workers accordingly)
#SBATCH --mem=32G                   # Memory per node
#SBATCH --time=01:00:00             # Maximum running time (HH:MM:SS)
#SBATCH --output=nwm_download_%j.out # Standard output file
#SBATCH --error=nwm_download_%j.err  # Standard error file

# Load necessary modules (adjust based on your HPC environment)
module load python/python3/3.11.7 # <--- UNCOMMENT this line if you are using system modules
# Install s3fs if not already available in your user environment for this module-loaded Python
pip install --user s3fs # <--- UNCOMMENT this line if you are using system modules and need s3fs
# -----------------------------------------------------------
#source /path/to/your/venv/bin/activate # If using a virtual environment

# Navigate to your script directory
# IMPORTANT: Replace '/path/to/your/scripts' with the actual path where you save download_nwm_hpc.py
cd /home/dlfernando/LDAS

# Define the range of years to download
# Load necessary modules (adjust based on your HPC environment)
module load python/python3/3.11.7 # <--- UNCOMMENT this line if you are using system modules
# Install s3fs if not already available in your user environment for this module-loaded Python
pip install --user s3fs # <--- UNCOMMENT this line if you are using system modules and need s3fs
# -----------------------------------------------------------
#source /path/to/your/venv/bin/activate # If using a virtual environment

# Navigate to your script directory
# IMPORTANT: Replace '/path/to/your/scripts' with the actual path where you save download_nwm_hpc.py
cd /home/dlfernando/LDAS

# Define the range of years to download
START_YEAR=1995
END_YEAR=2005

# Define other parameters
START_MONTH=1
END_MONTH=12
PRODUCT_PREFIX="LDASIN_DOMAIN1"
MAX_FILES=0 # 0 for all files per year
OUTPUT_BASE_DIR="/scratch/dlfernando/netcdf_hawii"
MAX_WORKERS=32 # Matched with --cpus-per-task

echo "Starting NWM data download for years ${START_YEAR} to ${END_YEAR}"
echo "Output base directory: ${OUTPUT_BASE_DIR}"

# Execute the Python script with the defined arguments for the entire year range
# The Python script itself will loop through the years
python hpc_download.py \
  --start_year ${START_YEAR} \
  --end_year ${END_YEAR} \
  --start_month ${START_MONTH} \
  --end_month ${END_MONTH} \
  --product_prefix ${PRODUCT_PREFIX} \
  --max_files ${MAX_FILES} \
  --output_dir ${OUTPUT_BASE_DIR} \
  --max_workers ${MAX_WORKERS}

echo "NWM data download job completed."
