#!/usr/bin/env python3
"""
Main script for processing mBAN data and generating segmented datasets.

This script is specifically designed to work with muscleBAN (mBAN) accelerometer data
from OpenSignals files. It processes the raw data, segments activities into sub-activities,
and saves the results as numpy files for further analysis.
It also creates a summary of the dataset.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
# external imports
import os
import sys
from pathlib import Path

# internal imports
from constants import MBAN_FS
from raw_data_processor import generate_mban_segmented_dataset, create_mban_dataset_summary

# ------------------------------------------------------------------------------------------------------------------- #
# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))



# ------------------------------------------------------------------------------------------------------------------- #
# configuration
# ------------------------------------------------------------------------------------------------------------------- #
# Enable/disable different processing steps
GENERATE_SEGMENTED_DATASET = True
CREATE_SUMMARY = True

# Define paths for raw data and output
RAW_DATA_FOLDER_PATH = '/Volumes/USB DISK/Backup PrevOccupAI data/Prevoccupai_HAR/subject_data/raw_signals_backups' 
OUTPUT_FOLDER_PATH = '/Volumes/USB DISK/Backup PrevOccupAI data/Prevoccupai_HAR/subject_data/new_segmented_data'     

# Processing parameters
CROP_SECONDS = 5           # Seconds to crop from beginning and end of each segment
PLOT_SEGMENTS = False      # Set to True ONLY for debugging specific files (will block execution!)
PLOT_CROPPED = False       # Set to True to visualize final cropped segments

# ------------------------------------------------------------------------------------------------------------------- #
# helper functions
# ------------------------------------------------------------------------------------------------------------------- #
def validate_paths():
    """Validate that the required paths exist and are accessible."""
    
    if not os.path.exists(RAW_DATA_FOLDER_PATH):
        print(f"Error: Raw data path does not exist: {RAW_DATA_FOLDER_PATH}")
        print("Please update RAW_DATA_FOLDER_PATH in this script to point to your data.")
        return False
    
    # Check for acquisitions folder or subject folders
    acquisitions_path = os.path.join(RAW_DATA_FOLDER_PATH, 'acquisitions')
    if os.path.exists(acquisitions_path):
        data_path = acquisitions_path
    else:
        data_path = RAW_DATA_FOLDER_PATH
    
    # Look for subject folders (folders starting with 'P')
    subject_folders = [f for f in os.listdir(data_path) 
                      if os.path.isdir(os.path.join(data_path, f)) and f.startswith('P')]
    
    if not subject_folders:
        print(f"Error: No subject folders found in {data_path}")
        print("Expected folders like P001, P002, etc.")
        return False
    
    print(f"Found {len(subject_folders)} subject folders: {subject_folders[:5]}{'...' if len(subject_folders) > 5 else ''}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
    
    return True


def print_mban_info():
    """Print information about mBAN processing."""
    
    print("=== mBAN Data Processing Configuration ===")
    print(f"Sampling frequency: {MBAN_FS} Hz")
    print(f"Crop seconds: {CROP_SECONDS}")
    print(f"Plot segments: {PLOT_SEGMENTS}")
    print(f"Plot cropped: {PLOT_CROPPED}")
    print(f"Raw data path: {RAW_DATA_FOLDER_PATH}")
    print(f"Output path: {OUTPUT_FOLDER_PATH}")
    print()
    
    if PLOT_SEGMENTS:
        print("⚠️  WARNING: PLOT_SEGMENTS is enabled!")
        print("   This will save segmentation plots for every file processed.")
        print("   For large datasets, consider setting PLOT_SEGMENTS = False")
        print("   to avoid creating hundreds of plot files.")
        print()
    
    print("Expected mBAN file format:")
    print("  opensignals_<MAC_ADDRESS>_<DATE>_<TIME>.txt")
    print("  Examples:")
    print("    opensignals_84FD27E506E8_2022-05-02_10-00-01.txt (mBAN sensor)")
    print("    opensignals_588E81A24A27_2022-05-02_10-00-01.txt (mBAN sensor)")
    print("  Note: Files with Android sensor names will be ignored:")
    print("    opensignals_ANDROID_ACCELEROMETER_2022-05-02_10-00-01.txt (ignored)")
    print("    opensignals_ANDROID_GYROSCOPE_2022-05-02_10-00-01.txt (ignored)")
    print()
    
    print("📊 Enhanced Segmentation Features:")
    print("  ✅ Subject-specific parameter optimization for problematic files")
    print("  ✅ Automatic post-processing for oversegmentation cases")  
    print("  ✅ Protocol-compliant segment validation")
    print("  ✅ Comprehensive error handling and file skipping")
    print()


# ------------------------------------------------------------------------------------------------------------------- #
# main execution
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    
    print("=== mBAN Data Processing Pipeline ===")
    print()
    
    # Print configuration info
    print_mban_info()
    
    # Validate paths
    if not validate_paths():
        print("Please fix the path configuration and try again.")
        sys.exit(1)
    
    # Step 1: Generate segmented dataset
    if GENERATE_SEGMENTED_DATASET:
        print("Step 1: Generating segmented mBAN dataset...")
        print("-" * 50)
        
        try:
            generate_mban_segmented_dataset(
                raw_data_path=RAW_DATA_FOLDER_PATH,
                output_path=OUTPUT_FOLDER_PATH,
                fs=MBAN_FS,
                crop_n_seconds=CROP_SECONDS,
                plot_segment_lines=PLOT_SEGMENTS,
                plot_cropped_tasks=PLOT_CROPPED
            )
            print("✓ Segmented dataset generation completed successfully!")
            
        except Exception as e:
            print(f"✗ Error during dataset generation: {str(e)}")
            print("Please check your data paths and file formats.")
            sys.exit(1)
    
    # Step 2: Create dataset summary
    if CREATE_SUMMARY:
        print("\nStep 2: Creating dataset summary...")
        print("-" * 50)
        
        try:
            create_mban_dataset_summary(OUTPUT_FOLDER_PATH)
            print("✓ Dataset summary completed!")
            
        except Exception as e:
            print(f"✗ Error creating summary: {str(e)}")
    
    print("\n=== Processing Complete ===")
    print(f"Check the output folder for results: {OUTPUT_FOLDER_PATH}")
    print()
    print("Next steps:")
    print("1. Review the generated numpy files in the 'segmented_data' folder")
    print("2. Use the data for feature extraction and machine learning")
    print("3. Modify the visualization settings if you want to inspect the segments")
