#!/bin/bash
#SBATCH --job-name=nwm_process    
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32        s
#SBATCH --mem=64G                 
#SBATCH --time=04:00:00           
#SBATCH --output=nwm_process_%j.out
#SBATCH --error=nwm_process_%j.err 

module load python/python3/3.11.7

pip install --user xarray pandas tqdm pyproj netcdf4 h5netcdf rioxarray

cd /home/dlfernando/LDAS

echo "Starting NWM data processing job..."

python process.py \
  --input_netcdf_dir /scratch/dlfernando/netcdf_hawii/1997\
  --pixel_data_file /home/dlfernando/LDAS/all_divide_pixeles_all.json \
  --output_csv_dir /scratch/dlfernando/processed_nwm_data \
  --variables LWDOWN PSFC Q2D SWDOWN T2D U2D V2D RAINRATE \
  --max_files 0 \
  --num_processes 32 # Match with --cpus-per-task for optimal usage

echo "NWM data processing job completed."
