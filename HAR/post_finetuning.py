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

# internal imports
from raw_data_processor.load_sensor_data import load_data_from_same_recording
from .load import load_labels_from_log
from .post_process import majority_vote_mid, threshold_tuning, heuristics_correction, expand_classification
from feature_extractor import pre_process_sensors
from constants import TXT


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def perform_post_processing(raw_data_path: str, label_map: Dict[str, int], min_durations: Dict[int, int],
                            fs: int, w_size: float, threshold: Optional[float] = None, load_sensors: Optional[List[str]] = None,
                            nr_samples_mv: int = 20) -> None:
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
        cfg = tsfel.load_json(f".\\HAR\\production_models\\{int(w_size*100)}_w_size\\cfg_file_production_model.json")
        features = tsfel.time_series_features_extractor(cfg, sensor_data, window_size=int(w_size*100), fs=fs, header_names=sensor_names)

        # load the model
        har_model, feature_names = _load_production_model(f".\\HAR\\production_models\\{int(w_size*100)}_w_size\\HAR_model_500.joblib")

        # get the features that are needed fot the classifier
        features = features[feature_names]

        # classify and post-process
        results_dict = _apply_post_processing(features, true_labels, har_model, w_size, fs,
                                                              nr_samples_mv, threshold, min_durations, subject)

        # add to dictionaty
        subject_predictions_dict[subject] = results_dict

    # put dictionary into a pandas dataframe
    results_df = pd.DataFrame.from_dict(subject_predictions_dict, orient='index')

    # save dataframe as csv file
    results_df.to_csv(f".\\HAR\\production_models\\{int(w_size*100)}_w_size\\post_processing_results.csv", index=True)


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
                           fs: int, nr_samples_mv: int, threshold: Optional[float], min_durations: Dict[int,int],
                           subject_id: str) -> Dict[str, float]:
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
    :return: Dict[str, float] where the keys is the post-processing scheme (name) and the value is the accuracy.
    """

    # list for holding the lists with the predictions
    predictions = []

    # list for holding accuracies
    accuracies = []

    # list for holding the expanded classifications
    expanded_predictions = []

    # classify the data - vanilla model
    y_pred = har_model.predict(features)

    # apply majority voting
    y_pred_mv = majority_vote_mid(y_pred, nr_samples_mv)

    # apply threshold tuning
    y_pred_proba = har_model.predict_proba(features)

    # check if a threshold was given as input
    if threshold is None:

        # find the best threshold
        best_threshold = _find_best_threshold(y_pred_proba, y_pred, labels, 0, 1, min_durations, w_size, fs)

        # use the best threshold for all post-processing methods
        threshold = best_threshold

        print(best_threshold)

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

    # add threshold to the dictionary
    results_dict['threshold_value'] = threshold

    # plot predictions and save
    _plot_all_predictions(labels, expanded_predictions, accuracies, post_processing_schemes, w_size, subject_id)

    return results_dict


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

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))  # Leave space for subtitle
    plt.savefig(f".\\HAR\\production_models\\{int(w_size*100)}_w_size\\post_processing_results_fig_{subject_id}.png")


def _find_best_threshold(probabilities: np.ndarray, y_pred: np.ndarray, true_labels: np.ndarray, sit_label: int,
                         stand_label: int, min_durations: Dict[int, float], w_size: float, fs: int) -> float:
    """
    Finds the best threshold value for post-processing, based on the accuracy of the threshold tuning + heuristics
    method. Tests the following threshold values: 0.6, 0.65, 0.7, 0.75, 0.8, 0.85.

    :param probabilities: np.ndarray of shape (n_samples, n_classes) containing the predicted probabilities
    :param y_pred: np.ndarray containing the predicted class labels (as integers)
    :param true_labels: np.ndarray containing the true labels
    :param sit_label: int corresponding to the sitting class label
    :param stand_label: int corresponding to the standing class label
    :param min_durations: Dictionary mapping each class label to its minimum segment duration in seconds.
    :param w_size: size of the window in seconds
    :param fs: the sampling frequency
    :return: float corresponding to the best threshold
    """

    # variable to store the highest accuracy
    best_acc = 0

    # variable for holding the best threshold
    best_threshold = 0

    # iterate through different threshold values
    for threshold in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:

        # adjust the prediction according to the threshold
        y_pred_tt = threshold_tuning(probabilities, y_pred, sit_label, stand_label, threshold)

        # combine with heuristics
        y_pred_tt_heur = heuristics_correction(y_pred_tt, w_size, min_durations)

        # expand classification to the size of the true label vector
        y_pred_tt_heur_expanded = expand_classification(y_pred_tt_heur, w_size, fs)
        y_pred_tt_expanded = expand_classification(y_pred_tt, w_size, fs)

        # calculate accuracy
        tt_heur_acc = accuracy_score(y_true=true_labels, y_pred=y_pred_tt_heur_expanded)
        tt_acc = accuracy_score(y_true=true_labels, y_pred=y_pred_tt_expanded)

        print(f"threshold: {threshold}")
        print(f"tt + heur: {tt_heur_acc}")

        # check if it's the highest accuracy and update variable
        if tt_heur_acc > best_acc:
            best_acc = tt_heur_acc
            best_threshold = threshold

    return best_threshold