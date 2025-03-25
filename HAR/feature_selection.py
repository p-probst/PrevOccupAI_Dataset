"""
Functions for loading model agnostic feature selection.

Available Functions
-------------------
[Public]
remove_low_variance(...): Applies Variance Thresholding to remove low-variance features.
remove_highly_correlated_features(...): Removes highly correlated features.
select_k_best_features(...): Select the top k best features based on their relationship with the target variable.

------------------
[Private]
None
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import pandas as pd
import numpy as np
from typing import Tuple


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def remove_low_variance(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies Variance Thresholding to remove low-variance features.

    :param X_train: Training features as a pandas DataFrame.
    :param X_test: Testing features as a pandas DataFrame.
    :param threshold: Variance threshold (default: 0.0, removes features with zero variance).
    :return: Tuple containing the transformed (X_train, X_test) DataFrames.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_train_selected = selector.fit_transform(X_train)  # Fit only on X_train
    X_test_selected = selector.transform(X_test)  # Apply transformation to X_test

    # Get the retained feature names
    retained_columns = X_train.columns[selector.get_support()]

    # Convert back to DataFrame
    X_train_filtered = pd.DataFrame(X_train_selected, columns=retained_columns, index=X_train.index)
    X_test_filtered = pd.DataFrame(X_test_selected, columns=retained_columns, index=X_test.index)

    return X_train_filtered, X_test_filtered


def remove_highly_correlated_features(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes highly correlated features from X_train and X_test.

    :param X_train: Training features as a pandas DataFrame.
    :param X_test: Testing features as a pandas DataFrame.
    :param threshold: Correlation threshold for feature removal (default: 0.9).
    :return: Tuple containing the transformed (X_train, X_test) DataFrames.
    """
    # Compute correlation matrix
    corr_matrix = X_train.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify features with correlation above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop correlated features
    X_train_filtered = X_train.drop(columns=to_drop)
    X_test_filtered = X_test.drop(columns=to_drop)

    return X_train_filtered, X_test_filtered


def select_k_best_features(X_train, X_test, y_train, k=10):
    """
    Select the top k best features based on their relationship with the target variable.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Testing feature set.
        y_train (pd.Series or np.array): Target values for training data.
        k (int): Number of best features to select.

    Returns:
        pd.DataFrame, pd.DataFrame: Transformed X_train and X_test with top k features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)  # Use f_regression for regression problems
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()]

    # Convert back to DataFrame
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)

    return X_train_selected, X_test_selected

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
