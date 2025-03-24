# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# internal imports
from constants import VALID_SENSORS, SEGMENTED_DATA_FOLDER, EXTRACTED_FEATRES_FOLDER, RANDOM_SEED
from raw_data_processor import generate_segmented_dataset
from feature_extractor import extract_features
from HAR import load_features, remove_low_variance, remove_highly_correlated_features, select_k_best_features
import numpy as np


# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
GENERATE_SEGMENTED_DATASET = False
EXTRACT_FEATURES = False
RF_HAR = True

# definition of folder_path
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

    if RF_HAR:

        print("HAR model training/test")

        # set random seed
        np.random.seed(RANDOM_SEED)

        # path to feature folder
        feature_data_folder = os.path.join(OUTPUT_FOLDER_PATH, EXTRACTED_FEATRES_FOLDER, "w_1-5_sc_none")

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

        # setup the models to be used
        random_forest = RandomForestClassifier()
        param_grid_rf = {"criterion": ['gini', 'entropy'],
                         "n_estimators": [50, 100],
                         "max_depth": [2, 4, 6]}

        # setup cross-validation
        inner_cv = GroupKFold(n_splits=2)

        # adding model specific feature selector on inner cv
        #feature_selector = RFE(estimator=random_forest, step=1, n_features_to_select=10, verbose=1)
        # feature_selector = SequentialFeatureSelector(estimator=random_forest, n_features_to_select='auto',
        #                                              direction='forward', cv=inner_cv)
        # pipeline = Pipeline([('feature_selection', feature_selector), ('classifier', random_forest)])

        # inner grid-search
        grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf,
                                   scoring='accuracy', n_jobs=-1, cv=inner_cv,
                                   verbose=1, refit=True)

        # list for holding the outer scores
        outer_scores = []

        # setup cross-validation for outer model comparison
        outer_cv = GroupKFold(n_splits=5)

        print('### ----------------------------------------- ###')
        print('Algorithm: Random Forest')

        # run outer loop
        for outer_train_idx, valid_idx in outer_cv.split(X_train, y_main_train, groups=subject_ids_train):

            # run inner loop for hyperparameter optimization
            grid_search.fit(X_train.iloc[outer_train_idx], y_main_train.iloc[outer_train_idx], groups=subject_ids_train.iloc[outer_train_idx])

            # print the best estimator and its accuracies
            print('\n-------------------------')
            print(f'Best accuracy (inner fold avg.): {grid_search.best_score_ * 100:.2f} %')
            print(f'best params: {grid_search.best_params_}')

            # # get the selected features
            # selected_features = X_train.columns[grid_search.best_estimator_.named_steps['feature_selection'].support_].tolist()
            # print(f'selected features: {selected_features}')

            # test the best estimator on the validation set of the current fold
            y_pred = grid_search.best_estimator_.predict(X_train.iloc[valid_idx])
            fold_score = accuracy_score(y_main_train.iloc[valid_idx], y_pred)
            print(f'--> fold accuracy: {fold_score * 100:.2f} %')

            # append outer score
            outer_scores.append(fold_score)

        # calculate average accuracy over all folds for the model
        avg_score = np.mean(outer_scores) * 100
        std_score = np.std(outer_scores) * 100

        print(f'\naverage score over all folds: {avg_score:.2f} +/- {std_score:.2f}')
        print('### ----------------------------------------- ###')



        print('testing')



