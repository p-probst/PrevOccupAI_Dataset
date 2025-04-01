# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from sklearn.model_selection import GroupShuffleSplit

# internal imports
from constants import VALID_SENSORS, SEGMENTED_DATA_FOLDER, EXTRACTED_FEATURES_FOLDER, RANDOM_SEED
from raw_data_processor import generate_segmented_dataset
from feature_extractor import extract_features
from HAR import (load_features, remove_low_variance, remove_highly_correlated_features, select_k_best_features,
                 evaluate_models, evaluate_production_model)
import numpy as np


# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
GENERATE_SEGMENTED_DATASET = False
EXTRACT_FEATURES = False
ML_HAR = True
ML_MODEL_SELECTION = True
ML_TRAIN_PRODUCTION_MODEL = False

# definition of folder_path
RAW_DATA_FOLDER_PATH = 'G:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\subject_data\\raw_signals_backups\\acquisitions'
OUTPUT_FOLDER_PATH = 'G:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\subject_data\\'

# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    if GENERATE_SEGMENTED_DATASET:

        # generate the segmented data set
        generate_segmented_dataset(RAW_DATA_FOLDER_PATH, OUTPUT_FOLDER_PATH, load_sensors=VALID_SENSORS,
                                   fs=100, output_file_type='.npy', plot_cropped_tasks=False, plot_segment_lines=False)

    if EXTRACT_FEATURES:

        print("extracting features")

        # path to segmented data folder
        segmented_data_path = os.path.join(OUTPUT_FOLDER_PATH, SEGMENTED_DATA_FOLDER)

        # extract features and save them to individual subject files
        extract_features(segmented_data_path, OUTPUT_FOLDER_PATH, window_scaler='standard',
                         default_input_file_type='.npy',
                         output_file_type='.npy')

    if ML_HAR:

        # setting variables for run
        # TODO: this has to be solved in a different way
        #norm_type = 'none'
        balancing_type = 'main_classes'
        #num_features_retain = 5

        print("HAR model training/test")

        # set random seed
        np.random.seed(RANDOM_SEED)

        for norm_type in ['none', 'minmax', 'standard']:

            # path to feature folder (change the folder name to run the different normalization schemes)
            feature_data_folder = os.path.join(OUTPUT_FOLDER_PATH, EXTRACTED_FEATURES_FOLDER, f"w_1-5_sc_{norm_type}")

            # load feature, labels, and subject IDs
            # (change balance_data='sub_classes' when wanting to classify all sub-classes individually)
            X, y_main, y_sub, subject_ids = load_features(feature_data_folder, balance_data=balancing_type)

            # split of train and test set
            splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_SEED)
            train_idx, test_idx = next(splitter.split(X, y_main, groups=subject_ids))

            print(f"subjects train: {subject_ids[train_idx].unique()}")
            print(f"subjects test: {subject_ids[test_idx].unique()}")

            # get train and test sets
            X_train_unp = X.iloc[train_idx]
            X_test_unp = X.iloc[test_idx]

            # get y depending on the balancing type
            if balancing_type == 'main_classes':
                y_train = y_main.iloc[train_idx]
                y_test = y_main.iloc[test_idx]

            else:  # sub-class balancing
                y_train = y_sub[train_idx]
                y_test = y_sub[test_idx]

            # get the subjects for training
            subject_ids_train = subject_ids.iloc[train_idx]

            for num_features_retain in [5, 10, 15, 20, 25, 30, 35]:

                print("\n.................................................................")
                print(f"Testing {num_features_retain} features with norm type \'{norm_type}\'...\n")

                # perform model agnostic feature selection
                X_train, X_test = remove_low_variance(X_train_unp, X_test_unp, threshold=0.1)
                X_train, X_test = remove_highly_correlated_features(X_train, X_test, threshold=0.9)
                X_train, X_test = select_k_best_features(X_train, X_test, y_train, k=num_features_retain)

                print(f"Used features: {X_train.columns.values}")

                if ML_MODEL_SELECTION:
                    print("Evaluating different models (Random Forest vs. KNN vs. SVM)")

                    # evaluate the models using main_class labels
                    evaluate_models(X_train, y_train, subject_ids_train, norm_type=norm_type)

                # if ML_TRAIN_PRODUCTION_MODEL:
                #     print("training and evaluating production model")
                #
                #     # evaluate production model
                #     evaluate_production_model(X_train, y_train, X_test, y_test, subject_ids_train, cv_splits=2)










