# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# internal imports
from constants import VALID_SENSORS, SEGMENTED_DATA_FOLDER, EXTRACTED_FEATRES_FOLDER, RANDOM_SEED
from raw_data_processor import generate_segmented_dataset
from feature_extractor import extract_features
from HAR import load_features, remove_low_variance, remove_highly_correlated_features, select_k_best_features, evaluate_models
import numpy as np


# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
GENERATE_SEGMENTED_DATASET = False
EXTRACT_FEATURES = False
ML_HAR = True
ML_MODEL_SELECTION = True
ML_TRAIN_PRODUCTION_MODEL = False

# definition of folder_path (change these paths to where you store the data)
RAW_DATA_FOLDER_PATH = 'D:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\subject_data\\raw_signals_backups\\acquisitions'
OUTPUT_FOLDER_PATH = 'D:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\subject_data\\'

# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    if GENERATE_SEGMENTED_DATASET:

        # generate the segmented data set
        generate_segmented_dataset(RAW_DATA_FOLDER_PATH, OUTPUT_FOLDER_PATH, load_sensors=VALID_SENSORS,
                                   fs=100, output_file_type='.csv', plot_cropped_tasks=False, plot_segment_lines=False)

    if EXTRACT_FEATURES:

        print("extracting features")

        # path to segmented data folder
        segmented_data_path = os.path.join(OUTPUT_FOLDER_PATH, SEGMENTED_DATA_FOLDER)

        # extract features and save them to individual subject files
        extract_features(segmented_data_path, OUTPUT_FOLDER_PATH, window_scaler=None, default_input_file_type='.npy',
                         output_file_type='.npy')

    if ML_HAR:

        print("HAR model training/test")

        # set random seed
        np.random.seed(RANDOM_SEED)

        # path to feature folder (change the folder name to run the different normalization schemes)
        feature_data_folder = os.path.join(OUTPUT_FOLDER_PATH, EXTRACTED_FEATRES_FOLDER, "w_1-5_sc_standard")

        # load feature, labels, and subject IDs
        X, y_main, y_sub, subject_ids = load_features(feature_data_folder, balance_data='main_classes')

        # split of train and test set
        splitter = GroupShuffleSplit(test_size=0.2, n_splits=1)
        train_idx, test_idx = next(splitter.split(X, y_main, groups=subject_ids))

        print(f"subjects train: {subject_ids[train_idx].unique()}")
        print(f"subjects test: {subject_ids[test_idx].unique()}")

        # get train and test sets
        X_train = X.iloc[train_idx]
        y_main_train = y_main.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_main_test = y_main.iloc[test_idx]

        subject_ids_train = subject_ids.iloc[train_idx]

        # (1) perform model agnostic feature selection
        X_train, X_test = remove_low_variance(X_train, X_test, threshold=0.1)
        X_train, X_test = remove_highly_correlated_features(X_train, X_test, threshold=0.9)
        X_train, X_test = select_k_best_features(X_train, X_test, y_main_train, k=10)

        if ML_MODEL_SELECTION:

            # evaluate the different models
            evaluate_models(X_train, y_main_train, subject_ids_train, norm_type='none')

        if ML_TRAIN_PRODUCTION_MODEL:

            print("training and evaluating production model")




        print('testing')



