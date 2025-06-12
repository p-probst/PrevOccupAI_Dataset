"""
Function for applying post-processing schemes to a prediction of a given classifier

Available Functions
-------------------
[Public]
perform_post_processing(...): Executes a post-processing pipeline for HAR using a pre-trained model

-------------------
[Private]
_load_production_model(...): Loads the pre-trained model
_pre_process_signals(...): Preprocesses the sensor data and label vector
_trim_data(...): Trims the sensor data to accommodate the full windowing
_apply_post_processing(...): Applies the post-processing schemes, implemented in post_process.py
_plot_all_predictions(...): Generates and saves a plot with the post-processing results for each subject
_find_best_threshold(...): Finds the threshold that produces the highest accuracy for threshold tuning + heuristics
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
import joblib
import tsfel
from typing import Optional, List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

# internal imports
from raw_data_processor.load_sensor_data import load_data_from_same_recording
from .load import load_labels_from_log
from .post_process import majority_vote_mid, threshold_tuning, heuristics_correction, expand_classification
from feature_extractor.feature_extractor import pre_process_sensors
from constants import TXT


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def perform_post_processing(raw_data_path: str, label_map: Dict[str, int], fs: int, w_size: float,
                            min_durations: Optional[Dict[int, int]] = None, threshold: Optional[float] = None,
                            load_sensors: Optional[List[str]] = None, nr_samples_mv: Optional[int] = None) -> None:
    """
    Executes a post-processing pipeline for Human Activity Recognition (HAR) using a pre-trained model.

    This function iterates over subject folders within the given directory, loads and preprocesses
    sensor data and corresponding labels, extracts features, classifies the data using a pre-trained HAR model,
    and applies post-processing logic to refine predictions. The results of the post-processing over all subjects
    are saved as a CSV file. This functions also generates and saves a plot for each subject with the results.

    :param raw_data_path: path to the folder containing the subject folders
    :param label_map: A dictionary mapping activity strings to numeric labels.
                        e.g., {"sitting": 1, "standing": 2, "walking": 3}.
    :param min_durations: Dictionary mapping each class label to its minimum segment duration in seconds.
    :param fs: the sampling frequency
    :param w_size: the window size in samples
    :param threshold: The probability margin threshold for adjusting predictions. Default is 0.1.
    :param load_sensors: list of sensors (as strings) indicating which sensors should be loaded. Default: None (all sensors are loaded)
    :param nr_samples_mv: the number of samples until the current position of the classifier
    :return: None
    """

    # list all folders within the raw_data_path
    subject_folders = os.listdir(raw_data_path)

    # get the folders that contain the subject data. Subject data folders start with 'S' (e.g., S001)
    subject_folders = [folder for folder in subject_folders if folder.startswith('S')]

    # dictionaty for holding the results for all subjects
    subject_predictions_dict = {}

    # dictionaty for holding the parameter optimization results for all subjects
    subject_opt_mv_results = {}
    subject_opt_tt_results = {}
    subject_opt_heur_results = {}

    for subject in subject_folders:

        print("\n#----------------------------------------------------------------------#")
        print(f"# Processing data for Subject {subject}")

        # get the path to the subject folder
        subject_folder_path = os.path.join(raw_data_path, subject)

        # list all files/folders inside the subject_folder
        folder_items = os.listdir(subject_folder_path)

        # get the folder containing the signals
        signals_path = [item for item in folder_items if os.path.isdir(os.path.join(subject_folder_path, item))]

        # path to the folder containing the data
        data_folder_path = os.path.join(subject_folder_path, signals_path[0])

        # get the txt file containing the labels
        txt_files = [item for item in folder_items if item.endswith(TXT)]

        # get the txt file path
        txt_file_path = os.path.join(subject_folder_path, txt_files[0])

        # load sensor data into a pandas dataframe
        subject_data = load_data_from_same_recording(data_folder_path, load_sensors, fs=fs)

        # generate label vector
        labels = load_labels_from_log(txt_file_path, label_map, subject_data.shape[0])

        # add the labels to the dataframe
        subject_data['labels'] = labels

        # get the sensor names
        sensor_names = subject_data.columns.values[1:-1]

        # pre process signals
        sensor_data, true_labels = _pre_process_signals(subject_data, sensor_names, w_size=w_size, fs=fs)

        # extract features
        cfg = tsfel.load_json(f".\\HAR\\production_models\\{int(w_size*fs)}_w_size\\cfg_file_production_model.json")
        features = tsfel.time_series_features_extractor(cfg, sensor_data, window_size=int(w_size*fs), fs=fs, header_names=sensor_names)

        # load the model
        har_model, feature_names = _load_production_model(f".\\HAR\\production_models\\{int(w_size*fs)}_w_size\\HAR_model_500.joblib")

        # get the features that are needed fot the classifier
        features = features[feature_names]

        # classify and post-process
        results_dict, opt_mv_results, opt_tt_results, opt_heur_results = _apply_post_processing(features, true_labels, har_model, w_size, fs,
                                                              nr_samples_mv, threshold, min_durations, subject)

        # add to dictionaty
        subject_predictions_dict[subject] = results_dict
        subject_opt_mv_results[subject] = opt_mv_results
        subject_opt_tt_results[subject] = opt_tt_results
        subject_opt_heur_results[subject] = opt_heur_results

    # put dictionary into a pandas dataframe
    results_df = pd.DataFrame.from_dict(subject_predictions_dict, orient='index')

    # save dataframe as csv file
    results_df.to_csv(f".\\HAR\\production_models\\{int(w_size*fs)}_w_size\\post_processing_results.csv", index=True)

    # if optimization was performed, save results
    if any(v is not None for v in subject_opt_mv_results.values()):
        pd.DataFrame.from_dict(subject_opt_mv_results, orient='index') \
            .to_csv(f".\\HAR\\production_models\\{int(w_size * fs)}_w_size\\opt_mv_results.csv", index=True)

    if any(v is not None for v in subject_opt_tt_results.values()):
        pd.DataFrame.from_dict(subject_opt_tt_results, orient='index') \
            .to_csv(f".\\HAR\\production_models\\{int(w_size * fs)}_w_size\\opt_tt_results.csv", index=True)

    if any(v is not None for v in subject_opt_heur_results.values()):
        pd.DataFrame.from_dict(subject_opt_heur_results, orient='index') \
            .to_csv(f".\\HAR\\production_models\\{int(w_size * fs)}_w_size\\opt_heur_results.csv", index=True)


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _load_production_model(model_path: str) -> Tuple[RandomForestClassifier, List[str]]:
    """
    Loads the production model
    :param model_path: path o the model
    :return: a tuple containing the model and the list of features used
    """
    # load the classifier
    har_model = joblib.load(model_path)

    # print model name
    print(f"model: {type(har_model).__name__}")
    print(f"\nhyperparameters: {har_model.get_params()}")

    # print the classes that the model saw during training
    print(f"\nclasses: {har_model.classes_}")

    # get the features that the model was trained with
    feature_names = har_model.feature_names_in_
    print(f"\nnumber of features: {len(feature_names)}")
    print(f"features: {feature_names}")

    return har_model, feature_names


def _pre_process_signals(subject_data: pd.DataFrame, sensor_names: List[str], w_size: float,
                         fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-processes the sensors contained in data_array according to their sensor type. Removes samples from the
    impulse response of the filters and trims the data and label vector to accommodate full windowing of the data.

    :param subject_data: pandas.DataFrame containing the sensor data
    :param sensor_names: list of strings correspondent to the sensor names
    :param w_size: window size in seconds
    :param fs: the sampling frequency
    :return: the processed sensor data and label vector
    """

    # convert data to numpy array
    sensor_data = subject_data.values[:,1:-1]

    # get the label vector
    labels = subject_data.values[:, -1]

    # pre-process the data
    sensor_data = pre_process_sensors(sensor_data, sensor_names)

    # remove impulse response
    sensor_data = sensor_data[250:,:]
    labels = labels[250:]

    # trim the data to accommodate full windowing
    sensor_data, to_trim = _trim_data(sensor_data, w_size=w_size, fs=fs)
    labels = labels[:-to_trim]

    return sensor_data, labels

def _trim_data(data: np.ndarray, w_size: float, fs: int) -> Tuple[np.ndarray, int]:
    """
    Function to get the amount that needs to be trimmed from the data to accommodate full windowing of the data
    (i.e., not excluding samples at the end).
    :param data: numpy.array containing the data
    :param w_size: Window size in seconds
    :param fs: Sampling rate
    :return: the trimmed data and the amount of samples that needed to be trimmed.
    """

    # calculate the amount that has to be trimmed of the signal
    to_trim = int(data.shape[0] % (w_size * fs))

    return data[:-to_trim, :], to_trim


def _apply_post_processing(features: np.ndarray, labels: np.ndarray, har_model: RandomForestClassifier, w_size: float,
                           fs: int, nr_samples_mv: Optional[int], threshold: Optional[float], min_durations: Optional[Dict[int,int]],
                           subject_id: str) -> Tuple[Dict[str, float], Optional[Dict[int, float]], Optional[Dict[float, float]], Optional[Dict[str, float]]]:
    """
    Applies multiple post-processing schemes to the predictions of a classifier (Random Forest): majority voting,
    threshold tuning, heuristics, threshold tuning + majority voting, and threshold tuning + heuristics. Check the
    documentation of these post-processing methods in post_process.py. This function generates a plot with the
    vanilla accuracy and the post-processing accuracies for each subject, as well as a .csv file containing the
    results for all subjects.

    :param features: numpy.array of shape (n_samples, n_features) containing the features
    :param labels: numpy.array containing the true class labels
    :param har_model: object from RandomForestClassifier
    :param w_size: window size in seconds
    :param fs: the sampling frequency
    :param nr_samples_mv: number of samples until the current position of the classifier
    :param threshold: The probability margin threshold for adjusting predictions. Default is 0.1.
    :param min_durations: Dictionary mapping each class label to its minimum segment duration in seconds.
    :param subject_id: str with the subject identifier
    :return: Dict[str, float] where the keys is the post-processing scheme (name) and the value is the accuracy.
    """

    # list for holding the lists with the predictions
    predictions = []

    # list for holding accuracies
    accuracies = []

    # list for holding the expanded classifications
    expanded_predictions = []

    # empty variables in case there is not optimization of the post-processing parameters
    optimization_results_mv = None
    optimization_results_tt = None
    optimization_results_heur = None

    # classify the data - vanilla model
    y_pred = har_model.predict(features)

    # get class probabilities
    y_pred_proba = har_model.predict_proba(features)

    if nr_samples_mv is None:

        # find best window size for majority voting
        best_window, optimization_results_mv = _optimize_majority_voting_window(y_pred, labels, w_size, fs)

        # update value
        nr_samples_mv = best_window

    if threshold is None:

        # find the best threshold
        best_threshold, optimization_results_tt = _optimize_threshold(y_pred_proba, y_pred, labels, 0, 1, w_size, fs)

        # use the best threshold for all post-processing methods
        threshold = best_threshold

    if min_durations is None:

        # find the best combination of durations
        best_durations, optimization_results_heur = _optimize_heuristics_parameters(y_pred, labels, w_size, fs)

        # update value
        min_durations = best_durations

    # apply majority voting
    y_pred_mv = majority_vote_mid(y_pred, nr_samples_mv)

    # apply threshold tuning
    y_pred_tt = threshold_tuning(y_pred_proba, y_pred,0,1, threshold)

    # apply heuristics
    y_pred_heur = heuristics_correction(y_pred, w_size, min_durations)

    # combine tt with mv
    y_pred_tt_mv = majority_vote_mid(y_pred_tt, 9)

    # combine tt with heuristics
    y_pred_tt_heur = heuristics_correction(y_pred_tt, w_size, min_durations)

    # append predictions to list
    predictions += [y_pred, y_pred_mv, y_pred_tt, y_pred_heur, y_pred_tt_mv, y_pred_tt_heur]

    # extend classifications and calculate the accuracies
    for prediction in predictions:

        # expand the predictions to the size of the original signal
        y_pred_expanded = expand_classification(prediction, w_size = w_size, fs=fs)
        expanded_predictions.append(y_pred_expanded)

        # calculate accuracy
        accuracy = accuracy_score(labels, y_pred_expanded)
        accuracies.append(round(accuracy*100, 2))


    # get a list containing the type of post-processing schemes
    post_processing_schemes = ["vanilla model acc", "majority voting acc", "threshold tuning acc", "heuristics acc",
                               "tt + mv acc", "tt + heur acc"]

    # save results to a dictionary
    results_dict = dict(zip(post_processing_schemes, accuracies))

    # plot predictions and save
    _plot_all_predictions(labels, expanded_predictions, accuracies, post_processing_schemes, w_size, subject_id)

    return results_dict, optimization_results_mv, optimization_results_tt, optimization_results_heur


def _plot_all_predictions(labels: np.ndarray, expanded_predictions: List[List[int]], accuracies: List[float],
                          post_processing_schemes: List[str], w_size: float, subject_id: str) -> None:
    """
    Generates and saves a figure with 6 plots. The first plot corresponds to true labels over time, and the other five plots
    correspond to the vanilla models and post-processing results over time.
    :param labels: numpy.array containing the true labels
    :param expanded_predictions: List os numpy.arrays containing the predictions expanded to the size of the true label vector
    :param accuracies: List containing the accuracies of the vanilla model and the 4 post-processing schemes.
    :param post_processing_schemes: list of strings pertaining to the name of the post-processing type
    :param w_size: window size in seconds
    :param subject_id: str with the subject identifier
    :return: None
    """
    n_preds = len(expanded_predictions)
    fig, axes = plt.subplots(nrows=n_preds + 1, ncols=1, sharex=True, sharey=True, figsize=(30, 3 * (n_preds + 1)))
    fig.suptitle(f"True labels vs post-processed predictions (Window size: {int(w_size*100)} samples)", fontsize=24)

    # Plot true labels
    axes[0].plot(labels, color='teal')
    axes[0].set_title("True Labels", fontsize=18)

    # Plot each prediction
    for i, (pred, acc, name) in enumerate(zip(expanded_predictions, accuracies, post_processing_schemes)):
        axes[i + 1].plot(pred, color='darkorange')
        axes[i + 1].set_title(f"{name}: {acc}%", fontsize=18)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    plt.savefig(f".\\HAR\\production_models\\{int(w_size*100)}_w_size\\post_processing_results_fig_{subject_id}.png")


def _optimize_threshold(probabilities: np.ndarray, y_pred: np.ndarray, true_labels: np.ndarray, sit_label: int,
                         stand_label: int, w_size: float, fs: int) -> Tuple[float, Dict[float, float]]:
    """
    Finds the best threshold value for post-processing, based on the accuracy of the threshold tuning
    method. Tests the following threshold values: 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95.

    :param probabilities: np.ndarray of shape (n_samples, n_classes) containing the predicted probabilities
    :param y_pred: np.ndarray containing the predicted class labels (as integers)
    :param true_labels: np.ndarray containing the true labels
    :param sit_label: int corresponding to the sitting class label
    :param stand_label: int corresponding to the standing class label
    :param w_size: size of the window in seconds
    :param fs: the sampling frequency
    :return: float corresponding to the best threshold
    """

    # variable to store the highest accuracy
    best_acc = 0

    # variable for holding the best threshold
    best_threshold = 0

    # tuple containing the different threshold values to test
    thresholds_tuple = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # list for holding the accuracies for each threshold tested
    acc_list = []

    # iterate through different threshold values
    for threshold in tqdm(thresholds_tuple, total=len(thresholds_tuple), desc="Optimizing threshold"):

        # adjust the prediction according to the threshold
        y_pred_tt = threshold_tuning(probabilities, y_pred, sit_label, stand_label, threshold)

        # expand classification to the size of the true label vector
        y_pred_tt_expanded = expand_classification(y_pred_tt, w_size, fs)

        # calculate accuracy
        tt_acc = accuracy_score(y_true=true_labels, y_pred=y_pred_tt_expanded)

        # append to list
        acc_list.append(round(tt_acc*100, 2))

        # check if it's the highest accuracy and update variable
        if tt_acc > best_acc:
            best_acc = tt_acc
            best_threshold = threshold

    # save results to the dict
    results_dict = dict(zip(thresholds_tuple, acc_list))

    return best_threshold, results_dict


def _optimize_majority_voting_window(y_pred: np.ndarray, true_labels: np.ndarray, w_size: float, fs: int) -> Tuple[int, Dict[int, float]]:

    # variable for holding the best accuracy
    best_acc = 0

    # variable for holding the best window
    best_window = 0

    # tuple with different window values to test
    windows_tuple = [5, 10, 15, 20]

    # list for holding the accuracies for each threshold tested
    acc_list = []

    # iterate through various window sizes
    for window in tqdm(windows_tuple, total=len(windows_tuple), desc="Optimizing majority voting window size"):

        # adjust the prediction with majority voting
        y_pred_mv = majority_vote_mid(y_pred, window)

        # expand classification to the size of the true label vector
        y_pred_mv_expanded = expand_classification(y_pred_mv, w_size, fs)

        # calculate accuracy
        mv_acc = accuracy_score(y_true=true_labels, y_pred=y_pred_mv_expanded)

        # append to accuracies list
        acc_list.append(round(mv_acc*100, 2))

        # check if it's the highest accuracy and update variable
        if mv_acc > best_acc:
            best_acc = mv_acc
            best_window = window

    # save results to the dict
    results_dict = dict(zip(windows_tuple, acc_list))

    return best_window, results_dict


def _optimize_heuristics_parameters(y_pred: np.ndarray, labels: np.ndarray, w_size: float, fs: int) \
                                    -> Tuple[Dict[int, int], Dict[str, float]]:

    # dictionary for holding the minimum durations for heuristics post-processing
    best_params = {}

    # variable for holding the highest accuracy
    best_acc  = 0

    # dictionary containing the durations to be tested for each class
    class_duration_test = {
        0: [20, 25, 30, 35, 40, 45],
        1: [15, 20, 25, 30],
        2: [5, 10, 15]
    }

    # list for holding the accuracies for each combination
    list_acc = []

    # list for holding each combination of durations as a Tuple (ex: (sitting duration, standing duration, walking duration) (30, 30, 10))
    list_combinations = []

    # Get list of class ids: [0, 1, 2]
    class_ids = list(class_duration_test.keys())

    # the lists of the durations to test for each class: [[20, 25, 30, 35, 40, 45], [20, 25, 30], [5, 10, 15]]
    durations_lists = [class_duration_test[c] for c in class_ids]

    # Calculate total number of iterations for tqdm
    total_combinations = 1
    for durations in durations_lists:
        total_combinations *= len(durations)

    # iterate through the various combinations of durations - avoids writing the 3 for loops
    for duration_list in tqdm(product(*durations_lists), total=total_combinations, desc="Optimizing heuristics"):

        # add combination list to the list
        list_combinations.append(duration_list)

        # update minimum durations dictionary
        min_durations = {class_id: dur for class_id, dur in zip(class_ids, duration_list)}

        # adjust the prediction with heuristics
        y_pred_heur = heuristics_correction(y_pred, w_size, min_durations)

        # expand  classification to the size of the label vector
        y_pred_heur_expanded = expand_classification(y_pred_heur, w_size, fs)

        # calculate accuracy
        heur_acc = accuracy_score(y_true=labels, y_pred=y_pred_heur_expanded)
        list_acc.append(round(heur_acc*100, 2))

        if heur_acc > best_acc:
            best_acc = heur_acc
            best_params = min_durations.copy()

    # save results to the dict
    results_dict = {"-".join(map(str, comb)): acc for comb, acc in zip(list_combinations, list_acc)}

    return best_params, results_dict