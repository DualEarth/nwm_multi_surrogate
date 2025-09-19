#!/bin/bash
#SBATCH --job-name=hawaii_download
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --qos=main 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          
#SBATCH --mem=32G                  
#SBATCH --time=01:00:00  
#SBATCH --output=nwm_download_%j.out
#SBATCH --error=nwm_download_%j.err 


module load python/python3/3.11.7 

pip install --user s3fs

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
MAX_WORKERS=32 

echo "Starting NWM data download for years ${START_YEAR} to ${END_YEAR}"
echo "Output base directory: ${OUTPUT_BASE_DIR}"

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
