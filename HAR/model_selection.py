"""
Function for model selection using traditional machine learning methods.

Available Functions
-------------------
[Public]
evaluate_models(...): Evaluates multiple machine learning models using nested cross-validation.

------------------
[Private]
_save_results(...): Saves the results from the nested cross-validation into a .csv file.
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# internal imports
from .cross_validation import nested_cross_val
from file_utils import create_dir

# ------------------------------------------------------------------------------------------------------------------- #
# constans
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
def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, subject_ids_train: pd.Series, norm_type: str) -> None:
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

    # init models depending on norm_type
    if norm_type == 'none':

        svm_dict = {ESTIMATOR: Pipeline([(STD_STEP, StandardScaler()), (SVM, SVC())]), PARAM_GRID: [
            {f'{SVM}__kernel': ['rbf'], f'{SVM}__C': np.power(10., np.arange(-4, 4)),
             f'{SVM}__gamma': np.power(10., np.arange(-5, 0))},
            {f'{SVM}__kernel': ['linear'], f'{SVM}__C': np.power(10., np.arange(-4, 4))}]},

        knn_dict = {ESTIMATOR: Pipeline([(STD_STEP, StandardScaler()), (KNN, KNeighborsClassifier(algorithm='ball_tree'))]),
                    PARAM_GRID: [{f'{KNN}__n_neighbors': list(range(1, 10)), f'{KNN}__p': [1, 2]}]}

    else:

        svm_dict = {ESTIMATOR: SVC(), PARAM_GRID: [
            {'kernel': ['rbf'], 'C': np.power(10., np.arange(-4, 4)), 'gamma': np.power(10., np.arange(-5, 0))},
            {'kernel': ['linear'], 'C': np.power(10., np.arange(-4, 4))}]},

        knn_dict = {ESTIMATOR: KNeighborsClassifier(algorithm='ball_tree'),
                    PARAM_GRID: [{f'{KNN}__n_neighbors': list(range(1, 10)), f'{KNN}__p': [1, 2]}]}

    # dict storing all different models
    model_dict = {

        KNN: knn_dict,

        SVM: svm_dict,

        RF: {ESTIMATOR: RandomForestClassifier(), PARAM_GRID: [
            {"criterion": ['gini', 'entropy'], "n_estimators": [50, 100, 500], "max_depth": [2, 5, 10, 20]}]},
    }

    for model_name, param_dict in model_dict.items():
        print('### ----------------------------------------- ###')
        print(f'Algorithm: {model_name}')

        # get the estimator and the param grid
        est = param_dict['estimator']
        param_grid_est = param_dict['param_grid']

        info_df = nested_cross_val(X_train, y_train, subject_ids_train, estimator=est, param_grid=param_grid_est)

        # save the results
        _save_results(info_df, estimator_name=model_name, norm_type=norm_type)


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _save_results(info_df: pd.DataFrame, estimator_name: str, norm_type: str) -> None:
    """
    Saves the results from the nested cross-validation into a .csv file.
    :param info_df: pandas.DataFrame containing the information from the nested cross-validation
    :param estimator_name: the name of the estimator used in the nested cross-validation
    :param norm_type: the normalization type used.
    :return: None
    """

    # get the path to the current project
    project_path = os.getcwd()

    # create results directory (if it doesn't exist)
    folder_path = create_dir(project_path, os.path.join("Results", "ML"))

    # create full file path
    file_path = os.path.join(folder_path, f'{estimator_name}_{norm_type}.csv')

    info_df.to_csv(file_path, sep=';', index=False)
