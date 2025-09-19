import s3fs
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_s3_file(s3_path, local_path, s3_filesystem):
    """
    Downloads a single file from S3 to a local path.

    Args:
        s3_path (str): The full S3 path to the file (e.g., "s3://bucket/key").
        local_path (str): The local file path where the S3 file will be saved.
        s3_filesystem (s3fs.S3FileSystem): An authenticated s3fs filesystem object.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        s3_filesystem.get(s3_path, local_path)
        print(f"Downloaded: {os.path.basename(s3_path)} to {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading {s3_path}: {e}")
        return False

def download_nwm_files(current_year, start_month, end_month, product_prefix,
                       max_files_to_download, output_directory, max_workers):
    """
    Downloads NWM retrospective data files for a given year and product.
    This function is adapted for non-interactive execution in an HPC environment.

    Args:
        current_year (int): The target year for which to download data.
        start_month (int): The starting month (inclusive, 1-12) for filtering files.
        end_month (int): The ending month (inclusive, 1-12) for filtering files.
        product_prefix (str): The prefix of the NWM product to download (e.g., "LDASIN_DOMAIN1").
        max_files_to_download (int or None): The maximum number of files to download.
                                              If None or 0, all matching files will be downloaded.
        output_directory (str): The local directory where files will be saved.
        max_workers (int): The maximum number of concurrent threads for downloading.
    """
    # Define the base S3 path for NWM retrospective data
    s3_base_path = f"s3://noaa-nwm-retrospective-3-0-pds/Hawaii/netcdf/FORCING/{current_year}/"

    # Initialize S3FileSystem for anonymous access to the NOAA PDS bucket
    # 'anon=True' for public buckets, 'region_name' for optimal performance
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs={'region_name': 'us-east-1'})

    # Create the output directory if it doesn't already exist
    # Appending the year to the output directory to organize files by year
    yearly_output_directory = os.path.join(output_directory, str(current_year))
    os.makedirs(yearly_output_directory, exist_ok=True)
    print(f"Ensured output directory exists: {yearly_output_directory}")

    print(f"Listing files in S3 path: {s3_base_path}...")
    try:
        # List all files (objects) within the specified S3 path
        all_s3_paths = s3.ls(s3_base_path)
        print(f"Found {len(all_s3_paths)} total items in S3.")
    except Exception as e:
        print(f"Error listing files from S3: {e}")
        return

    # Filter files based on product prefix, year, and month range
    filtered_files = []
    print(f"Filtering files for product '{product_prefix}' between months {start_month} and {end_month} for year {current_year}...")
    for s3_path in all_s3_paths:
        filename = os.path.basename(s3_path) # Extract just the filename

        # Check if the filename starts with the year and contains the product prefix
        if filename.startswith(str(current_year)) and f".{product_prefix}" in filename:
            try:
                # Extract month from filename (e.g., "YYYYMMDDHHmm.product_prefix...")
                # Assuming format like 200602010000.LDASIN_DOMAIN1.comp.h5
                # Month is typically at index 4 (0-indexed) and 2 characters long
                file_month = int(filename[4:6])
                if start_month <= file_month <= end_month:
                    filtered_files.append(f"s3://noaa-nwm-retrospective-3-0-pds/Hawaii/netcdf/FORCING/{current_year}/{filename}") # Reconstruct full S3 path
            except ValueError:
                # Skip files where month extraction fails
                print(f"Warning: Could not parse month from filename: {filename}. Skipping.")
                continue
    print(f"Found {len(filtered_files)} files matching filter criteria for year {current_year}.")

    # Apply the max_files_to_download limit if specified
    files_to_download = filtered_files
    if max_files_to_download is not None and max_files_to_download > 0:
        files_to_download = filtered_files[:max_files_to_download]
        print(f"Limiting download to the first {len(files_to_download)} files as requested for year {current_year}.")

    if not files_to_download:
        print(f"No files found matching the criteria or download limit for year {current_year}. Moving to next year (if any).")
        return

    print(f"Commencing download of {len(files_to_download)} files for year {current_year} using {max_workers} concurrent workers.")

    # Use ThreadPoolExecutor for concurrent downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_s3_file,
                s3_path,
                os.path.join(yearly_output_directory, os.path.basename(s3_path)),
                s3
            ): s3_path # Map future back to its original s3_path for reporting
            for s3_path in files_to_download
        }

        # Monitor and report on completed downloads
        for i, future in enumerate(as_completed(futures)):
            s3_path_completed = futures[future]
            if not future.result():
                print(f"Failed to download: {s3_path_completed}")
            # Optional: Add progress indicator
            # print(f"Progress: {i + 1}/{len(files_to_download)} files processed for year {current_year}.")

    print(f"\nBatch download complete for year {current_year}.")

if __name__ == "__main__":
    # Set up argument parsing for command-line execution
    parser = argparse.ArgumentParser(
        description="Download NWM retrospective data from S3, optimized for HPC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values
    )

    parser.add_argument(
        "--start_year",
        type=int,
        default=1994, # Updated default start year
        help="The starting year for NWM data download (inclusive)."
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=2013, # Updated default end year
        help="The ending year for NWM data download (inclusive)."
    )
    parser.add_argument(
        "--start_month",
        type=int,
        default=2,
        help="The starting month (1-12) for filtering files (inclusive)."
    )
    parser.add_argument(
        "--end_month",
        type=int,
        default=12,
        help="The ending month (1-12) for filtering files (inclusive)."
    )
    parser.add_argument(
        "--product_prefix",
        type=str,
        default="LDASIN_DOMAIN1",
        help="The product prefix to filter files (e.g., 'LDASIN_DOMAIN1')."
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0, # 0 means no limit
        help="Maximum number of files to download per year. Set to 0 or leave blank for all matching files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/dlfernando/netcdf_hawii", # Updated default output directory
        help="The base local directory to save the downloaded files. Subdirectories will be created per year."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of concurrent download threads."
    )

    args = parser.parse_args()

    # Convert max_files to None if it's 0, aligning with the function's logic
    max_download_limit = args.max_files if args.max_files > 0 else None

    print(f"\nStarting NWM Hawaii data download (HPC version) for years {args.start_year}-{args.end_year}...")
    print(f"  Months: {args.start_month}-{args.end_month}")
    print(f"  Product: {args.product_prefix}")
    print(f"  Max Files per Year: {'All' if max_download_limit is None else max_download_limit}")
    print(f"  Base Output Directory: {args.output_dir}")
    print(f"  Max Workers: {args.max_workers}")
    print("-" * 50)

    # Loop through the specified range of years
    for year_to_download in range(args.start_year, args.end_year + 1):
        print(f"\n--- Processing Year: {year_to_download} ---")
        download_nwm_files(
            current_year=year_to_download,
            start_month=args.start_month,
            end_month=args.end_month,
            product_prefix=args.product_prefix,
            max_files_to_download=max_download_limit,
            output_directory=args.output_dir, # Pass the base directory
            max_workers=args.max_workers
        )
        print("-" * 50)

    print("\nProgram finished. All specified years processed.")
