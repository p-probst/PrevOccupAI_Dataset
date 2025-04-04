"""
Functions for performing cross-validation for model selection and hyperparameter tuning.

Available Functions
-------------------
[Public]
nested_cross_val(...): Performs nested cross-validation using GroupKFold to evaluate a machine learning model with hyperparameter tuning.
tune_production_model(...): Performs hyperparameter tuning using GroupKFold, X and y.
------------------
[Private]
None
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from typing import Dict, Any, List, Union

from constants import RANDOM_SEED


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def nested_cross_val(X: pd.DataFrame, y: pd.Series, subject_ids: pd.Series, estimator: ClassifierMixin,
                     param_grid: Union[List[Dict[str, Any]], Dict[str, Any]],
                     splits_outer: int = 5, splits_inner: int = 2) -> pd.DataFrame():
    """
    Performs nested cross-validation using GroupKFold to evaluate a machine learning model with hyperparameter tuning.

    The function applies a nested cross-validation strategy, where the outer loop performs model evaluation
    and the inner loop performs hyperparameter tuning using GridSearchCV. The grouping variable `subject_ids`
    ensures that data from the same subject does not appear in both training and validation sets.

    The function generates a pandas.DataFrame containing the hyperparameters of the model as well as the avg. accuracy
    of the inner GridSearch, the accuracy per outer fold, and the mean accuracy and std over all folds.

    :param X: pandas.DataFrame containing the feature matrix
    :param y: pandas.Series containing the target labels
    :param subject_ids: pandas.Series containing the subject IDs.
    :param estimator: the estimator to be evaluated
    :param param_grid: the parameter grid for the estimator
    :param splits_outer: number of outer cross-validation splits. Default: 5
    :param splits_inner: number of inner cross-validation splits. Default: 2
    :return: pandas.DataFrame containing the best hyperparameters found for each outer fold,
        inner cross-validation accuracy, outer fold accuracy, and the overall mean and standard deviation
        of accuracy across all outer folds.
    """

    # lists for holding the scores of the outer folds and the info of each fold
    fold_scores = []
    fold_info = []

    # set up cross-validation for outer loop
    outer_cv = GroupKFold(n_splits=splits_outer, shuffle=True, random_state=RANDOM_SEED)

    # set up inner cross-validation for hyperparameter tuning using gridSearch
    inner_cv = GroupKFold(n_splits=splits_inner, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy',
                               n_jobs=-1, cv=inner_cv, verbose=1, refit=True)

    # run outer loop
    for fold_num, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y, groups=subject_ids), start=1):

        # get the corresponding X, y, and subject
        X_train = X.iloc[train_idx]
        X_val = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[valid_idx]

        # get the subject IDs for the inner cv
        subject_ids_inner_cv = subject_ids.iloc[train_idx]

        # run inner loop for hyperparameter optimization
        grid_search.fit(X_train, y_train, groups=subject_ids_inner_cv)

        # test the best estimator on the validation set of the current fold (this is possible as refit=True)
        y_pred = grid_search.best_estimator_.predict(X_val)
        fold_acc = accuracy_score(y_val, y_pred) * 100

        # print the results
        inner_acc = grid_search.best_score_ * 100
        best_params = grid_search.best_params_

        # print the results
        print(f'\nResults of fold {fold_num}/{splits_outer}')
        print(f'best accuracy (inner fold avg.): {inner_acc:.2f} %')
        print(f'best params: {best_params}')
        print(f'--> fold accuracy: {fold_acc:.2f} %')
        print('-------------------------')

        # append fold score
        fold_scores.append(fold_acc)

        # append the info for the fold
        best_params.update({'inner_acc': inner_acc, 'fold_acc': fold_acc})
        fold_info.append(best_params)

    # calculate the average accuracy over all outer folds
    outer_fold_avg_acc = np.mean(fold_scores)
    outer_fold_std = np.std(fold_scores)

    # print the result
    print(f'\naverage score over all folds: {outer_fold_avg_acc:.2f} +/- {outer_fold_std:.2f}')
    print('### ----------------------------------------- ###')

    # generate a pandas DataFrame containing the info
    info_df = pd.DataFrame(fold_info)

    # add the average score
    # (adding here to not have it to re-calculate again. The entire column thus has the same value)
    info_df['estimator_avg_acc'] = outer_fold_avg_acc
    info_df['estimator_std_acc'] = outer_fold_std

    return info_df


def tune_production_model(X: pd.DataFrame, y: pd.Series, subject_ids: pd.Series, estimator: ClassifierMixin,
                          param_grid: Union[List[Dict[str, Any]], Dict[str, Any]],
                          cv_splits: int = 5):
    """
    Performs hyperparameter tuning using GroupKFold, X and y. This function can be used to train the production model
    on the entire training set.
    :param X: pandas.DataFrame containing the feature matrix
    :param y: pandas.Series containing the target labels
    :param subject_ids: pandas.Series containing the subject IDs.
    :param estimator: the estimator to be evaluated
    :param param_grid: the parameter grid for the estimator
    :param cv_splits: the number of cross-validation splits for the gridsearch. Default: 5
    :return:
    """

    # set up cross-validation for hyperparameter tuning
    cv = GroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy',
                               n_jobs=1, cv=cv, verbose=1, refit=True)

    # perform hyperparameter tuning
    grid_search.fit(X, y, groups=subject_ids)

    # print the results
    print("\nResults of hyperparameter tuning")
    print(f"best score (CV avg.): {grid_search.best_score_ * 100}")
    print(f"best parameters: {grid_search.best_params_}")
    print('-------------------------')

    return grid_search.best_estimator_


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
