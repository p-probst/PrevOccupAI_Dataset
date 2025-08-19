from feature_extractor import extract_mban_features

if __name__ == '__main__':
    extract_mban_features(
        data_path='/Volumes/USB DISK/Backup PrevOccupAI data/Prevoccupai_HAR/subject_data/exported_segments_npy',         # Folder with subject folders (e.g., P001, P002, ...)
        features_data_path='/Volumes/USB DISK/Backup PrevOccupAI data/Prevoccupai_HAR/subject_data/output_features',    # Where to save extracted features
        activities=['sitting', 'walking', 'cabinets', 'stairs', 'standing'],  # Activities to process
        fs=1000,                                          # Sampling rate for mBAN
        window_size=5,                                  # Window size in seconds, changed from 0.5s to 5s
        overlap=0.5,                                      # Window overlap (0.5 = 50%)
        window_scaler='minmax',                           # Optional: 'minmax', 'standard', or None
        output_file_type='.npy'                           # Output format: '.npy' or '.csv'
    )
