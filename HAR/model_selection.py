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
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# internal imports
from .cross_validation import nested_cross_val, tune_production_model
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

    model_dict = {

         KNN: {ESTIMATOR: Pipeline([(STD_STEP, StandardScaler()), (KNN, KNeighborsClassifier(algorithm='ball_tree'))]),
               PARAM_GRID: [{f'{KNN}__n_neighbors': list(range(1, 15)), f'{KNN}__p': [1, 2]}]},

         SVM: {ESTIMATOR: Pipeline([(STD_STEP, StandardScaler()), (SVM, SVC())]), PARAM_GRID: [
             {f'{SVM}__kernel': ['rbf'], f'{SVM}__C': np.power(10., np.arange(-4, 4)),
              f'{SVM}__gamma': np.power(10., np.arange(-5, 0))},
             {f'{SVM}__kernel': ['linear'], f'{SVM}__C': np.power(10., np.arange(-4, 4))}]},

         RF: {ESTIMATOR: RandomForestClassifier(), PARAM_GRID: [
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

def evaluate_production_model(X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series, subject_ids_train: pd.Series) -> None:
    """

    :param X_train: pandas.DataFrame containing the training data
    :param y_train: pandas.Series containing the training labels
    :param X_test: pandas.DataFrame containing the test data
    :param y_test: pandas.Series containing the training labels
    :param subject_ids_train: pandas.Series containing the subject IDs
    :return: None
    """

    # define estimator and hyperparameters
    estimator = RandomForestClassifier()

    param_grid = {"criterion": ['gini', 'entropy'],
                  "n_estimators": [50, 100, 500, 1000],
                  "max_depth": [2, 5, 10, 20, 30]}

    # perform hyperparameter tuning
    model = tune_production_model(X=X_train, y=y_train, subject_ids=subject_ids_train,
                                  estimator=estimator, param_grid=param_grid, cv_splits=2)

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

    # save model
    # get the project path
    project_path = os.getcwd()
    model_path = os.path.join(project_path, "HAR", "HAR_model.joblib")
    joblib.dump(model, model_path)



# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
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

    # create results directory (if it doesn't exist)
    folder_path = create_dir(project_path, os.path.join("Results", "ML"))

    # create full file path
    file_path = os.path.join(folder_path, f'{estimator_name}_cl{num_classes}_f{num_features}_wNorm-{norm_type}.csv')

    info_df.to_csv(file_path, sep=';', index=False)
