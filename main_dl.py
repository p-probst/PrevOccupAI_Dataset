
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import torch
print(torch.cuda.is_available())

from HAR.dl.train_test import plot_performance_history
# internal imports
from constants import SEGMENTED_DATA_FOLDER, MAIN_ACTIVITY_LABELS, SENSOR_COLS_JSON, LOADED_SENSORS_KEY
from HAR.dl import generate_dataset, get_train_test_data, select_idle_gpu, run_model_training
from HAR.dl import DL_DATASET
from HAR.dl import HARLstm
from file_utils import create_dir, load_json_file

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
GENERATE_DATASET = False
TRAIN_TEST_MODEL = True


# definition of folder_path
OUTPUT_FOLDER_PATH = 'D:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\subject_data'

# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    if GENERATE_DATASET:

        print("generating dataset for deep learning")

        # path to segmented data folder
        segmented_data_path = os.path.join(OUTPUT_FOLDER_PATH, SEGMENTED_DATA_FOLDER)

        generate_dataset(segmented_data_path, OUTPUT_FOLDER_PATH, window_size=0.5, default_input_file_type='.npy')

    if TRAIN_TEST_MODEL:

        # set window size, dataset path, and model save path
        window_size_samples = 100
        dataset_path = os.path.join(OUTPUT_FOLDER_PATH, DL_DATASET, f'w_{window_size_samples}')
        load_sensors = ["ACC", "GYR"]

        # define path json file containing the sensor columns
        numpy_columns_file = os.path.join(OUTPUT_FOLDER_PATH, SEGMENTED_DATA_FOLDER, SENSOR_COLS_JSON)
        # get sensor_columns
        sensor_columns = load_json_file(numpy_columns_file)[LOADED_SENSORS_KEY]

        # set number of epochs
        num_epochs = 30
        dropout = 0.3

        # set the GPU
        cuda_device = select_idle_gpu()

        print("training/testing model on generated dataset")
        train_dataloader, test_dataloader, num_channels = get_train_test_data(dataset_path, batch_size=64,
                                                          load_sensors=load_sensors, sensor_columns=sensor_columns,
                                                          norm_method="z-score", norm_type="subject",
                                                          balancing_type='main_classes')

        # set model variables and parameters
        # TODO: implement strategy to select only specific sensors
        har_model = HARLstm(num_features=num_channels, hidden_size=128, num_layers=2,
                            num_classes=len(MAIN_ACTIVITY_LABELS), dropout=dropout)

        # get the model name
        model_name = f"{har_model.__class__.__name__}_hs{har_model.hidden_size}_nl{har_model.num_layers}_do{int(dropout*100)}"

        # put model on cuda device
        har_model.to(cuda_device)

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(har_model.parameters(), lr=1e-4)

        # define path to save model
        model_save_path = create_dir(os.getcwd(),
                                     os.path.join("HAR", "dl", f"trained_models_w_size_{window_size_samples}", "_".join(load_sensors)))


        # run the training loop
        performance_history = run_model_training(model=har_model, model_save_path=model_save_path,
                                                 train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                                 criterion=criterion, optimizer=optimizer, cuda_device=cuda_device,
                                                 num_epochs=num_epochs, patience=20)

        # plot the performance history
        plot_performance_history(performance_dict=performance_history, model_name=model_name, save_path=model_save_path)

        # save the model history as CSV
        performance_df = pd.DataFrame(performance_history)
        performance_df.to_csv(os.path.join(model_save_path, f"{model_name}_performance.csv"))


