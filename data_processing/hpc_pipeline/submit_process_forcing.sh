#!/bin/bash
#SBATCH --job-name=nwm_process    # Name of your job
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32        # Requesting 16 cores
#SBATCH --mem=64G                 # Increase memory for large datasets if needed
#SBATCH --time=04:00:00           # Adjust max time for processing (e.g., 4 hours)
#SBATCH --output=nwm_process_%j.out # Standard output file
#SBATCH --error=nwm_process_%j.err  # Standard error file

# --- IMPORTANT: If you encounter "DOS line breaks" error for THIS SCRIPT, run this command: ---
# dos2unix submit_process_job.sh
# ------------------------------------------------------------------------------------------------

# --- CHOOSE ONE METHOD FOR PYTHON ENVIRONMENT SETUP BELOW ---
# IMPORTANT: Only uncomment ONE of the following methods for your Python environment.
# Do NOT uncomment both sections.

# METHOD 1: Using system Python modules
module load python/python3/3.11.7

# Install necessary Python packages for the processing script.
# This 'pip install --user' command only needs to be run ONCE per environment setup.
# If packages are already installed, pip will indicate it.
echo "Checking for required package installations and installing if necessary (Method 1)..."
pip install --user xarray pandas tqdm pyproj netcdf4 h5netcdf rioxarray
echo "Package installation check complete."

# METHOD 2: Using a Miniconda/Anaconda environment
# Uncomment and adjust paths if you prefer Miniconda.
# Remember to replace 'your_env_name' with your actual conda environment name.
# source /home/dlfernando/LDAS/bin/activate
# conda activate your_env_name # e.g., 'base' or 'nwm_env'
# Ensure packages are installed in your conda environment:
# (your_env_name) conda install xarray pandas tqdm pyproj netcdf4 h5netcdf rioxarray
# echo "Miniconda environment 'your_env_name' activated (Method 2)."


# Navigate to your script directory
cd /home/dlfernando/LDAS

echo "Starting NWM data processing job..."

# Execute the Python script with desired arguments
# Note: In your error, the script was named 'process.py'.
# Please ensure the filename matches what you use: 'process_nwm_data_hpc.py' or 'process.py'
python process.py \
  --input_netcdf_dir /scratch/dlfernando/netcdf_hawii/1997\
  --pixel_data_file /home/dlfernando/LDAS/all_divide_pixeles_all.json \
  --output_csv_dir /scratch/dlfernando/processed_nwm_data \
  --variables LWDOWN PSFC Q2D SWDOWN T2D U2D V2D RAINRATE \
  --max_files 0 \
  --num_processes 32 # Match with --cpus-per-task for optimal usage

echo "NWM data processing job completed."
