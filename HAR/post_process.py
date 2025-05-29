"""
Functions for post-processing the results of the prediction of a classifier

Available Functions
-------------------
[Public]
majority_vote_mid(...): Adjusts the prediction for a classifier using majority voting.
threshold_tuning(...): Adjusts the prediction for a classifier, based on  given threshold, by reducing the confusion between the 'sit' and 'stand' classes.
expand_classification(...): Expands the array containing the class predictions to the size of the original signal.

------------------
[Private]
_correct_short_segments(...) Replaces segments of a specific class that are shorter than a given duration. Replaces with the neighboring most frequent class.
_find_class_segments(...) Finds start and end indices of continuous segments belonging to the target class.
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from typing import Union, List, Dict, Tuple
from collections import Counter

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def majority_vote_mid(predictions: List[int], num_samples: int) -> List[int]:
    """
    Implementation of the majority vote (MV) according to Engelhardt and Hudgins as presented in:
    'A Robust, Real-Time Control Scheme for Multifunction Myoelectric Control'

    The majority vote decision d_mv for a sample d_i is calculated using the previous m samples and the next m samples.
    This means, that the MV is always calculated at the middle of the MV window (of size 2m +1) and thus a delay T_d
    would be produced  in a real-time system. When a constraint (regarding to maximum delay time) is put on the system:

                                tau*m <= T_d; with tau = processing delay (classification time)

    However these constraints do not apply to this function since the calculations are done offline (at the moment)

    :param predictions: list containing the predictions of the classifier
    :param num_samples: the number of samples until the current position of the classifier
    :return: list with the majority vote corrected predictions
    """

    # calculate the size of the MV window
    MV_window_length = 2 * num_samples + 1

    # get the number of predictions made by the classifier
    num_predictions = len(predictions)

    # define indices for start, end and middle MV window
    start = 0
    stop = MV_window_length
    mid = start + num_samples

    # calculate the number of iterations through the list
    steps = 1 + (num_predictions - MV_window_length)

    # copy the predicitons list for inserting the MV corrected values
    mv_corrected_predictions = predictions.copy()

    # cycle through the list
    for step in range(0, steps):

        # get the slice from the predictions
        mv_window = predictions[start:stop]

        # calculate majority vote
        MV, _ = Counter(mv_window).most_common(1)[0]

        # insert the mv corrected prediction at the position in the list
        mv_corrected_predictions[mid] = MV

        # update start, stop and middle
        start += 1
        stop += 1
        mid += 1

    return mv_corrected_predictions


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


def heuristics_correction(predictions: np.ndarray,
                                     window_size: float,
                                     min_durations: Dict[int, float]) -> np.ndarray:
    """
    Apply post-processing to correct short activity segments for each class.

    :param predictions: 1D array of predicted class labels.
    :param window_size: Duration of each prediction window in seconds.
    :param min_durations: Dictionary mapping each class label to its minimum segment duration in seconds.
    :return: Post-processed prediction array with short segments corrected.
    """
    corrected = predictions.copy()

    # Apply correction for each class using the specified minimum duration
    for class_id, min_duration in min_durations.items():
        corrected = _correct_short_segments(corrected, class_id, min_duration, window_size)

    return corrected


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


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _correct_short_segments(predictions: np.ndarray, class_id: int, min_duration: float, window_size: float) -> np.ndarray:
    """
    Replace segments of a specific class that are shorter than a given duration.

    The replacement class is chosen as the most frequent class from neighboring values.

    :param predictions: 1D array of predicted class labels.
    :param class_id: Class to check for short-duration segments.
    :param min_duration: Minimum acceptable duration for a segment in seconds.
    :param window_size: Duration of each prediction window in seconds.
    :return: Updated prediction array with short segments replaced.
    """

    # get the segments for the class
    segments = _find_class_segments(predictions, class_id)
    corrected = predictions.copy()

    # cycle over the segments
    for start, end in segments:

        # calculate the segment lengths in seconds
        duration = (end - start + 1) * window_size

        # check whether the segment needs to be corrected (too short)
        if duration < min_duration:
            # Get neighbor classes
            left = predictions[start - 1] if start > 0 else None
            right = predictions[end + 1] if end < len(predictions) - 1 else None

            neighbors = [c for c in (left, right) if c is not None]
            if neighbors:

                # in case the left and the right neighbor are from different
                # classes then the left neighbor (i.e., the previous activity - chronologically) is chosen
                replacement = Counter(neighbors).most_common(1)[0][0]
                corrected[start:end + 1] = replacement

    return corrected


def _find_class_segments(predictions: np.ndarray, target_class: int) -> List[Tuple[int, int]]:
    """
    Find start and end indices of contiguous segments belonging to the target class.

    :param predictions: 1D array of predicted class labels.
    :param target_class: Class label to find segments for.
    :return: List of tuples, where each tuple is (start_idx, end_idx) of a segment.
    """

    # list for holding the segments
    segments = []

    # init variables
    in_segment = False
    start = 0

    # cycle over the predictions
    for i, pred in enumerate(predictions):

        # check whether current prediciton corresponds to the target class
        if pred == target_class:

            # start counting the segment
            if not in_segment:
                in_segment = True
                start = i
        else:
            # end the segment
            if in_segment:
                segments.append((start, i - 1))
                in_segment = False

    # when reaching the end, assign the end of the prediction array as the stop
    if in_segment:
        segments.append((start, len(predictions) - 1))

    return segments