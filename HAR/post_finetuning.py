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

def perform_post_processing(raw_data_path: str, label_map: Dict[str, int], min_durations,
                            fs: int, w_size: int, threshold, load_sensors: Optional[List[str]] = None, nr_samples_mv: int = 20):

    # list all folders within the raw_data_path
    subject_folders = os.listdir(raw_data_path)

    # get the folders that contain the subject data. Subject data folders start with 'S' (e.g., S001)
    subject_folders = [folder for folder in subject_folders if folder.startswith('S')]

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
        subject_data, labels = _pre_process_signals(subject_data, sensor_names, w_size=w_size, fs=fs)

        # extract features
        cfg = tsfel.load_json(".\\HAR\\production_models\\500_w_size\\cfg_file_production_model.json")
        features = tsfel.time_series_features_extractor(cfg, subject_data, window_size=w_size, fs=fs, header_names=sensor_names)

        # load the model
        har_model, feature_names = _load_production_model(".\\HAR\\production_models\\500_w_size\\HAR_model_500.joblib")

        # get the features that are needed fot the classifier
        features = features[feature_names]

        # classify and post-process
        results_dict = _apply_post_processing(features, labels, har_model, w_size, fs, nr_samples_mv, threshold, min_durations)

        pass




# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _load_production_model(model_path: str) -> Tuple[RandomForestClassifier, List[str]]:

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


def _pre_process_signals(subject_data: pd.DataFrame, sensor_names, w_size: int, fs: int):

    # convert data to numpy array
    sensor_data = subject_data.values[:,1:-1]

    # get the label vector
    labels = subject_data.values[1:-1]

    # pre-process the data
    sensor_data = pre_process_sensors(sensor_data, sensor_names)

    # remove impulse response
    sensor_data = sensor_data[250:,:]
    labels = labels[250:]

    # trim the data to accomodate full windowing
    sensor_data, to_trim = _trim_data(sensor_data, w_size, fs)
    labels= labels[:-to_trim]

    return sensor_data, labels

def _trim_data(data, w_size, fs):
    """
        Function to get the amount that needs to be trimmed from the data to accomodate full windowing of the data (i.e., not excluding samples at the end).
        :param data: numpy.array containing the data
        :param w_size: Window size in seconds
        :param fs: Sampling rate
        :return: DataFrame containing the trimmed muscleBAN data.
        """

    # calculate the amount that has to be trimmed of the signal
    to_trim = int(data.shape[0] % (w_size * fs))

    return data[:-to_trim, :], to_trim


def _apply_post_processing(features: np.ndarray, labels: np.ndarray, har_model: RandomForestClassifier, w_size: int, fs: int, nr_samples_mv, threshold, min_durations):

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
        y_pred_expanded = expand_classification(prediction, w_size, fs)
        expanded_predictions.append(y_pred_expanded)

        # calculate accuracy
        accuracy = accuracy_score(labels, y_pred_expanded)
        accuracies.append(accuracy)

    # plot predictions

    # get a list containing the type of post-processing schemes
    post_processing_schemes = ["vanilla model", "majority voting", "threshold tuning", "heuristics", "tt + mv", "tt + heur"]

    # save results to a dictionary
    results_dict = dict(zip(post_processing_schemes, accuracies))

    return results_dict


def _plot_all_predictions(labels: np.ndarray, expanded_predictions, accuracies, post_processing_schemes, w_size):

        n_preds = len(expanded_predictions)
        fig, axes = plt.subplots(nrows=n_preds + 1, ncols=1, sharex=True, sharey=True, figsize=(30, 3 * (n_preds + 1)))
        fig.suptitle(f"True labels vs post-processed predictions (Window size: {w_size})", fontsize=24)

        # Plot true labels
        axes[0].plot(labels, color='teal')
        axes[0].set_title("True Labels", fontsize=18)

        # Plot each prediction
        for i, (pred, acc, name) in enumerate(zip(expanded_predictions, accuracies, post_processing_schemes)):
            axes[i + 1].plot(pred, color='darkorange')
            axes[i + 1].set_title(f"{name}: {acc * 100:.2f}%", fontsize=18)

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))  # Leave space for subtitle
        plt.show()