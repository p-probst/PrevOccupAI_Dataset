# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import pandas as pd
import joblib
import tsfel
from typing import Optional, List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier


# internal imports
from raw_data_processor.load_sensor_data import load_data_from_same_recording
from .load import load_labels_from_log
from .post_process import majority_vote_mid
from feature_extractor import pre_process_sensors
from constants import TXT


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def perform_post_processing(raw_data_path: str, fs: int, w_size: int, label_map: Dict[str, int], json_file_path: str,
                            model_path: str, load_sensors: Optional[List[str]] = None, nr_samples_mv: int = 20):

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
        subject_data, labels = _pre_process_signals(subject_data, w_size=w_size, fs=fs)

        # extract features
        cfg = tsfel.load_json(json_file_path)
        features = tsfel.time_series_features_extractor(cfg, subject_data, window_size=w_size, fs=fs, header_names=sensor_names)

        # load the model
        har_model, feature_names = _load_production_model(model_path)

        # get the features that are needed fot the classifier
        features = features[feature_names]

        # classify the data - vanilla model
        y_pred = har_model.predict(features)

        # post-processing
        y_pred_mv = majority_vote_mid(y_pred, nr_samples_mv)




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