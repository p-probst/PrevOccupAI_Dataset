"""
Functions for post-processing the results of the classification

Available Functions
-------------------
[Public]
threshold_tuning(...): Adjusts the prediction for a classifier, based on  given threshold, by reducing the confusion between the 'sit' and 'stand' classes.
expand_classification(...): Expands the array containing the class predictions to the size of the original signal.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from typing import Union, List

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def threshold_tuning(probabilities: np.ndarray, y_pred: Union[np.ndarray, list],
                     sit_label: int = 0, stand_label: int = 1, threshold: float = 0.1) -> np.ndarray:
    """
    Adjusts predictions for a classifier by reducing confusion between 'stand' and 'sit'.

    If the model predicts 'stand' (class 1) and the difference in predicted probability
    between 'stand' and 'sit' (class 0) is less than the given threshold, the prediction
    is changed to 'sit'.

    :param probabilities: numpy.array of shape (n_samples, n_classes) containing the predicted probabilities
    :param y_pred: array containing the predicted class labels (as integers)
    :param sit_label: Class label for 'sit'. Default is 0.
    :param stand_label: Class label for 'stand'. Default is 1.
    :param threshold: The probability margin threshold for adjusting predictions. Default is 0.1.
    :return: numpy.ndarray containing the adjusted class label predictions
    """
    adjusted = []
    for i, probs in enumerate(probabilities):
        pred = y_pred[i]

        # Apply adjustment only if model predicted 'stand'
        if pred == stand_label:
            p_stand = probs[stand_label]
            p_sit = probs[sit_label]
            if (p_stand - p_sit) < threshold:
                pred = sit_label  # Change prediction to 'sit'

        adjusted.append(pred)

    return np.array(adjusted)


def expand_classification(clf_result: List[int], w_size: int, fs: int) -> List[int]:
    """
    Converts the time column from the android timestamp which is in nanoseconds to seconds.
    Parameters.
    :param clf_result: list with the classifier prediction where each entry is the prediction made for a window.
    :param w_size: the window size in samples that was used to make the classification.
    :param fs: the sampling frequency of the signal that was classified.
    :return: the expanded classification results.
    """

    expanded_clf_result = []

    # cycle over the classification results list
    for i, p in enumerate(clf_result):
        expanded_clf_result += [p] * int(w_size * fs)

    return expanded_clf_result