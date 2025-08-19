#!/usr/bin/env python3
"""
Example script for using the mBAN feature extraction system.

This script demonstrates how to extract features from your segmented mBAN data.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from feature_extractor import extract_mban_features


def main():
    """Main function to extract features from mBAN data."""
    
    print("🔬 mBAN Feature Extraction")
    print("=" * 50)
    
    # Define paths - Update these paths to match your setup
    segmented_data_path = "/Volumes/USB DISK/Backup PrevOccupAI data/Prevoccupai_HAR/subject_data/new_segmented_data/segmented_data"
    output_path = "/Users/goncalobarros/Documents/projects/PrevOccupAI_Dataset/mban_features"
    
    # Feature extraction parameters
    activities = None  # None = all activities, or specify list like ['cabinets', 'sitting', 'walking']
    fs = 1000  # mBAN sampling rate
    window_size = 0.5  # 0.5 second windows (500 samples)
    overlap = 0.5  # 50% overlap
    window_scaler = None  # Options: None, 'minmax', 'standard'
    output_file_type = '.npy'  # Options: '.npy', '.csv'
    
    print(f"📁 Input path: {segmented_data_path}")
    print(f"📁 Output path: {output_path}")
    print(f"🎯 Activities: {'All' if activities is None else activities}")
    print(f"⚡ Sampling rate: {fs} Hz")
    print(f"⏱️ Window size: {window_size} seconds ({int(window_size * fs)} samples)")
    print(f"🔄 Overlap: {int(overlap * 100)}%")
    print(f"📊 Window scaling: {window_scaler if window_scaler else 'None'}")
    print(f"💾 Output format: {output_file_type}")
    print("=" * 50)
    
    # Check if input path exists
    if not os.path.exists(segmented_data_path):
        print(f"❌ Input path not found: {segmented_data_path}")
        print("Please mount the USB drive or adjust the path.")
        return
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    try:
        print("🔄 Starting feature extraction...")
        
        # Extract features from mBAN data
        extract_mban_features(
            data_path=segmented_data_path,
            features_data_path=output_path,
            activities=activities,
            fs=fs,
            window_size=window_size,
            overlap=overlap,
            window_scaler=window_scaler,
            output_file_type=output_file_type
        )
        
        print("\n✅ Feature extraction completed successfully!")
        print("\n📊 Output Structure:")
        print(f"  📁 {output_path}/")
        print("    📁 extracted_features/")
        print(f"      📁 w_{int(window_size*fs)}_sc_{'none' if window_scaler is None else window_scaler}/")
        print("        📄 P001.npy")
        print("        📄 P002.npy")
        print("        📄 ...")
        print("        📄 class_instances.json")
        
    except Exception as e:
        print(f"❌ Error during feature extraction: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
