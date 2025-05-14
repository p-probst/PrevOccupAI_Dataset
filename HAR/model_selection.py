"""
Function for model selection using traditional machine learning methods.

Available Functions
-------------------
[Public]
perform_model_selection(...): Evaluates 3 different models (Random Forest, KNN, and SVM)  using a nested cross-validation to select which of these models is used for production.
train_production_model(...): Trains the production model by performing a final hyperparameter tuning. The trained model is then evaluated on the test set.

------------------
[Private]
_evaluate_models(...): Evaluates multiple machine learning models using nested cross-validation.
_evaluate_production_model(...): Performs a final hyperparameter tuning and performs production model evaluation.
_save_results(...): Saves the results from the nested cross-validation into a .csv file.
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

# internal imports
from .load import load_features
from .cross_validation import nested_cross_val, tune_production_model
from .feature_selection import remove_low_variance, remove_highly_correlated_features, select_k_best_features
from constants import RANDOM_SEED
from file_utils import create_dir

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
RF = "Random Forest"
SVM = "SVM"
KNN = "KNN"

STD_STEP = 'std'
PARAM_GRID = 'param_grid'
ESTIMATOR = 'estimator'


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def perform_model_selection(data_path: str, balancing_type: str) -> None:
    """
    Evaluates 3 different models (Random Forest, KNN, and SVM)  using a nested cross-validation to select which of
    these models is used for production. The model selection is only performed on the training data.
    :param data_path: the path to the data. This should point to the folder containing the extracted features.
    :param balancing_type: the data balancing type. Can be either:
                         'main_classes': for balancing the data in such a way that each main class has the (almost) the
                                       same amount of data. This ensures that each sub-class within the main class has
                                       the same amount of instances.
                         'sub_classes': for balancing that all sub-classes have the same amount of instances
                         None: no balancing applied. Default: None
    :return: None
    """

    for norm_type in ['none', 'minmax', 'standard']:

        print(f'norm_type: {norm_type}')

        # TODO: @Sara set use the window size in samples as variable in this f-string
        # path to feature folder
        feature_data_folder = os.path.join(data_path, f"w_150_sc_{norm_type}")

        # load feature, labels, and subject IDs
        X, y_main, y_sub, subject_ids = load_features(feature_data_folder, balance_data=balancing_type)

        # split of train and test set
        splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_SEED)
        train_idx, test_idx = next(splitter.split(X, y_main, groups=subject_ids))

        print(f"subjects train: {subject_ids[train_idx].unique()}")
        print(f"subjects test: {subject_ids[test_idx].unique()}")

        # get train and test sets
        X_train_all_features = X.iloc[train_idx]
        # TODO: @Sara Please print here the total number of instances for training

        # get y depending on the balancing type
        if balancing_type == 'main_classes':
            y_train = y_main.iloc[train_idx]

        else:  # sub-class balancing
            y_train = y_sub[train_idx]

            # add label encoding, as in this case the labels are non-consecutive
            le = LabelEncoder()
            y_train = pd.Series(le.fit_transform(y_train))

        # get the subjects for training
        subject_ids_train = subject_ids.iloc[train_idx]

        for num_features_retain in [5, 10, 15, 20, 25, 30, 35]:
            print("\n.................................................................")
            print(f"Classes used: {np.unique(y_train)}")
            print(f"Testing {num_features_retain} features with norm type \'{norm_type}\'...\n")

            # perform model agnostic feature selection
            X_train, _ = remove_low_variance(X_train_all_features, X_test=None, threshold=0.1)
            X_train, _ = remove_highly_correlated_features(X_train, X_test=None, threshold=0.9)
            X_train, _ = select_k_best_features(X_train, y_train, X_test=None, k=num_features_retain)

            print(f"Used features: {X_train.columns.values}")

            # TODO: @Sara the window_size_samples has to be passed here as well to create the folder for storing the results
            # evaluate the models using main_class labels
            _evaluate_models(X_train, y_train, subject_ids_train, norm_type=norm_type)


def train_production_model(data_path: str, num_features_retain: int, balancing_type: str, norm_type: str) -> None:
    """
    Trains the production model by performing a final hyperparameter tuning. The trained model is then evaluated on the
    test set.
    :param data_path: the path to the data. This should point to the folder containing the extracted features.
    :param num_features_retain: the number of features to retain after model agnostic feature selection.
    :param balancing_type: the data balancing type. Can be either:
                         'main_classes': for balancing the data in such a way that each main class has the (almost) the
                                       same amount of data. This ensures that each sub-class within the main class has
                                       the same amount of instances.
                         'sub_classes': for balancing that all sub-classes have the same amount of instances
                         None: no balancing applied. Default: None
    :param norm_type: the normalization type used on the windowed data. Can either be 'minmax', 'std', or 'none'
    :return: None
    """

    # TODO: @Sara set use the window size in samples as variable in this f-string
    # path to feature folder (change the folder name to run the different normalization schemes)
    feature_data_folder = os.path.join(data_path, f"w_150_sc_{norm_type}")

    # load feature, labels, and subject IDs
    X, y_main, y_sub, subject_ids = load_features(feature_data_folder, balance_data=balancing_type)

    # split of train and test set
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y_main, groups=subject_ids))

    # get train and test sets
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    # TODO: @Sara Please print here the total number of instances for training

    print(f"subjects train: {subject_ids[train_idx].unique()}")
    print(f"subjects test: {subject_ids[test_idx].unique()}")

    # get y depending on the balancing type
    if balancing_type == 'main_classes':
        y_train = y_main.iloc[train_idx]
        y_test = y_main.iloc[test_idx]

    else:  # sub-class balancing
        y_train = y_sub[train_idx]
        y_test = y_sub[test_idx]

        # add label encoding, as in this case the labels are non-consecutive
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train))
        y_test = pd.Series(le.fit_transform(y_test))

    # get the subjects for training
    subject_ids_train = subject_ids.iloc[train_idx]

    # perform model agnostic feature selection
    X_train, X_test = remove_low_variance(X_train, X_test, threshold=0.1)
    X_train, X_test = remove_highly_correlated_features(X_train, X_test, threshold=0.9)
    X_train, X_test = select_k_best_features(X_train, y_train, X_test, k=num_features_retain)

    print(f"Classes used: {np.unique(y_train)}")
    print(f"Used features: {X_train.columns.values}")

    # TODO: @Sara the window_size_samples has to be passed here as well to save the model
    # evaluate production model
    _evaluate_production_model(X_train, y_train, X_test, y_test, subject_ids_train, cv_splits=2)

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, subject_ids_train: pd.Series, norm_type: str) -> None:
    """
    Evaluates multiple machine learning models using nested cross-validation.

    This function initializes and evaluates different models (SVM, KNN, and Random Forest)
    based on the specified normalization type. It applies nested cross-validation to assess
    model performance and saves the results.

    :param X_train: pandas.DataFrame containing the training data
    :param y_train: pandas.Series containing the labels
    :param subject_ids_train: pandas.Series containing the subject IDs
    :param norm_type: the normalization type used on the windowed data. Can either be 'minmax', 'std', or 'none'
    :return: None
    """

    model_dict = {

         KNN: {ESTIMATOR: Pipeline([(STD_STEP, StandardScaler()), (KNN, KNeighborsClassifier(algorithm='ball_tree'))]),
               PARAM_GRID: [{f'{KNN}__n_neighbors': list(range(1, 15)), f'{KNN}__p': [1, 2]}]},

         SVM: {ESTIMATOR: Pipeline([(STD_STEP, StandardScaler()), (SVM, SVC(random_state=RANDOM_SEED))]), PARAM_GRID: [
             {f'{SVM}__kernel': ['rbf'], f'{SVM}__C': np.power(10., np.arange(-4, 4)),
              f'{SVM}__gamma': np.power(10., np.arange(-5, 0))},
             {f'{SVM}__kernel': ['linear'], f'{SVM}__C': np.power(10., np.arange(-4, 4))}]},

         RF: {ESTIMATOR: RandomForestClassifier(random_state=RANDOM_SEED), PARAM_GRID: [
             {"criterion": ['gini', 'entropy'], "n_estimators": [50, 100, 500, 1000], "max_depth": [2, 5, 10, 20, 30]}]}
    }

    for model_name, param_dict in model_dict.items():
        print('### ----------------------------------------- ###')
        print(f'Algorithm: {model_name}')

        # get the estimator and the param grid
        est = param_dict[ESTIMATOR]
        param_grid_est = param_dict[PARAM_GRID]

        info_df = nested_cross_val(X_train, y_train, subject_ids_train, estimator=est, param_grid=param_grid_est)

        # save the results
        _save_results(info_df, estimator_name=model_name, num_classes=len(y_train.unique()),
                      num_features=len(X_train.columns), norm_type=norm_type)


def _evaluate_production_model(X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series, subject_ids_train: pd.Series,
                              cv_splits: int = 5) -> None:
    """
    Performs a final hyperparameter tuning on the production model that was chosen based on the results from
    evaluate_models(...) and evaluates the model on the test data.
    :param X_train: pandas.DataFrame containing the training data
    :param y_train: pandas.Series containing the training labels
    :param X_test: pandas.DataFrame containing the test data
    :param y_test: pandas.Series containing the training labels
    :param subject_ids_train: pandas.Series containing the subject IDs
    :param cv_splits: the number of cross-validation splits for the gridsearch. Default: 5
    :return: None
    """

    # define estimator and hyperparameters
    estimator = RandomForestClassifier(random_state=RANDOM_SEED)

    param_grid = {"criterion": ['gini', 'entropy'],
                  "n_estimators": [50, 100, 500, 1000],
                  "max_depth": [2, 5, 10, 20, 30]}

    # perform hyperparameter tuning
    model = tune_production_model(X=X_train, y=y_train, subject_ids=subject_ids_train,
                                  estimator=estimator, param_grid=param_grid, cv_splits=cv_splits)

    # get train and test accuracy
    train_acc = accuracy_score(y_true=y_train, y_pred=model.predict(X_train))
    test_acc = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))

    print("\nResults on production model evaluation.")
    print(f"train accuracy: {train_acc * 100: .2f}")
    print(f"test accuracy: {test_acc * 100: .2f}")

    # plot confusion matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix | Test set")
    plt.show()

    # TODO: @Sara please implement saving of the confusion matrix through code. They should be saved into the same
    #  folder as the model

    # TODO: @Sara here the window_size_samples has to be added to the model name. Use a f-string for that.
    #  Please create also a folder that stores the models. This way they are neatly organized. The folder name can be
    #  "trained_models". It should be within the "HAR" folder". Maybe a sub-folder where the model and its confusion
    #  matrix are stored is necessary
    # save model
    # get the project path
    project_path = os.getcwd()
    model_path = os.path.join(project_path, "HAR", "HAR_model.joblib")
    joblib.dump(model, model_path)


def _save_results(info_df: pd.DataFrame, estimator_name: str, num_classes: int, num_features: int, norm_type: str) -> None:
    """
    Saves the results from the nested cross-validation into a .csv file.
    :param info_df: pandas.DataFrame containing the information from the nested cross-validation
    :param estimator_name: the name of the estimator used in the nested cross-validation
    :param num_features: the number of features used
    :param norm_type: the normalization type used.
    :return: None
    """

    # get the path to the current project
    project_path = os.getcwd()

    # TODO: @Sara here the folder that has the window size as name should be added.
    #  It could be something like os.path.join("Results", "ML",f"{window_size_samples}", f"num_classes_{num_classes}"
    # create results directory (if it doesn't exist)
    folder_path = create_dir(project_path, os.path.join("Results", "ML", f"num_classes_{num_classes}"))

    # create full file path
    file_path = os.path.join(folder_path,f'{estimator_name}_f{num_features}_wNorm-{norm_type}.csv')

    info_df.to_csv(file_path, sep=';', index=False)
