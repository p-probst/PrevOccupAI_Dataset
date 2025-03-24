"""
Functions for performing cross-validation for model selection and hyperparameter tuning.

Available Functions
-------------------
[Public]

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
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from typing import Dict, Any, List, Union


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def nested_cross_val(X: pd.DataFrame, y: pd.Series, subject_ids: pd.Series, estimator: ClassifierMixin,
                     param_grid: Union[List[Dict[str, Any]], Dict[str, Any]],
                     splits_outer: int = 5, splits_inner: int = 2) -> pd.DataFrame():
    """

    :param X:
    :param y:
    :param subject_ids:
    :param estimator:
    :param param_grid:
    :param splits_outer:
    :param splits_inner:
    :return:
    """

    # lists for holding the scores of the outer folds and the info of each fold
    fold_scores = []
    fold_info = []

    # set up cross-validation for outer loop
    outer_cv = GroupKFold(n_splits=splits_outer)

    # set up inner cross-validation for hyperparameter tuning using gridSearch
    inner_cv = GroupKFold(n_splits=splits_inner)
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
        fold_acc = accuracy_score(y_val, y_pred)

        # print the results
        inner_acc = grid_search.best_score_
        best_params = grid_search.best_params_

        # print the results
        print(f'\nResults of fold {fold_num}/{splits_outer}')
        print(f'best accuracy (inner fold avg.): {inner_acc * 100:.2f} %')
        print(f'best params: {best_params}')
        print(f'--> fold accuracy: {fold_acc * 100:.2f} %')
        print('-------------------------')

        # append fold score
        fold_scores.append(fold_acc)

        # append the info for the fold
        best_params.update({'inner_acc': inner_acc, 'fold_acc': fold_acc})
        fold_info.append(best_params)

    # calculate the average accuracy over all outer folds
    outer_fold_avg_acc = np.mean(fold_scores) * 100
    outer_fold_std = np.std(fold_scores) * 100

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


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #