import os
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd



print(f"CUDA available: {torch.cuda.is_available()}")


# internal imports
from constants import SEGMENTED_DATA_FOLDER, MAIN_ACTIVITY_LABELS, SENSOR_COLS_JSON, LOADED_SENSORS_KEY, VALID_SENSORS, RANDOM_SEED
from HAR.dl import generate_dataset, get_train_test_data, run_model_training, select_idle_gpu, configure_seed
from HAR.dl import DL_DATASET
from HAR.dl import HARRnn
from HAR.dl.train_test import plot_performance_history
from HAR.post_processing_optimizer import dl_optimize_post_processing
from file_utils import create_dir, load_json_file

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
GENERATE_DATASET = False
TRAIN_TEST_MODEL = False
DL_POST_PROCESSING = True

DRIVE = "F"

# definition of folder_path
OUTPUT_FOLDER_PATH = f'{DRIVE}:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\subject_data'

# ------------------------------------------------------------------------------------------------------------------- #
# argument parsing
# ------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()

# (1) paths
parser.add_argument('--data_path', default=OUTPUT_FOLDER_PATH, help="Path to dataset.")

# (2) dataset
parser.add_argument('--fs', default=100, type=int, help="The sampling frequency used during data acquisition.")
parser.add_argument('--window_size_s', default=5, type=float, help="The window size (in seconds) for sequence windowing. Can be either int or float.")
parser.add_argument('--seq_len', default=10, type=int, help="The window size (in samples) for sub-sequencing input samples. Should be a factor of window_size (in samples).")
parser.add_argument('--load_sensors', nargs="+", default=None, help="The sensor to be loaded (as List[str]), e.g., [\"ACC\", \"GYR\"].")
parser.add_argument('--norm_method', default='z-score', choices=['z-score', 'min-max', None], help="The normalization method (as str) used.")
parser.add_argument('--norm_type', default='subject', choices=['global', 'subject', 'window', None], help="The type of normalization (as str).")
parser.add_argument('--balancing_type', default='main_classes', choices=['main_classes', 'sub_classes', None], help="The balancing type (as str).")

# (3) model related parameters
parser.add_argument('--model_type', default='lstm', type=str, help="The model to be trained", choices=['lstm', 'gru'])
parser.add_argument('--num_epochs', default=40, type=int, help="The number of epochs used in model training.")
parser.add_argument('--batch_size', default=64, type=int, help="The batch size used in model training.")
parser.add_argument('--hidden_size', default=128, type=int, help="The hidden size used in RNN models (LSTM, GRU).")
parser.add_argument('--num_layers', default=1, type=int, help="The number of layers used in RNN models (LSTM, GRU).")
parser.add_argument('--dropout', default=0.3, type=float, help="The dropout rate used during model training.")
parser.add_argument('--lr', default=1e-4, type=float, help="The learning rate used during model training")

# parse the provided arguments
parsed_args = parser.parse_args()


# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    # obtain window size and sampling frequency
    window_size = parsed_args.window_size_s
    fs = parsed_args.fs

    # get window size in samples
    window_size_samples = int(window_size * fs)

    # check whether a dataset for the provided window_size has already been generated
    if not os.path.exists(os.path.join(OUTPUT_FOLDER_PATH, DL_DATASET, f"w_{window_size_samples}")):

        print("generating dataset for deep learning")

        # path to segmented data folder
        segmented_data_path = os.path.join(OUTPUT_FOLDER_PATH, SEGMENTED_DATA_FOLDER)

        generate_dataset(segmented_data_path, OUTPUT_FOLDER_PATH, window_size=window_size, default_input_file_type='.npy')

    if TRAIN_TEST_MODEL:

        # set dataset related parameters
        dataset_path = os.path.join(OUTPUT_FOLDER_PATH, DL_DATASET, f'w_{window_size_samples}')
        load_sensors = parsed_args.load_sensors
        seq_len = parsed_args.seq_len
        norm_method = parsed_args.norm_method
        norm_type = parsed_args.norm_type
        balancing_type = parsed_args.balancing_type

        # check whether None was passed for load_sensors
        if not load_sensors: load_sensors = VALID_SENSORS

        # set model related parameters
        model_type = parsed_args.model_type
        num_epochs = parsed_args.num_epochs
        batch_size = parsed_args.batch_size
        hidden_size = parsed_args.hidden_size
        num_layers = parsed_args.num_layers
        dropout = parsed_args.dropout
        lr = parsed_args.lr

        # set the GPU
        cuda_device = select_idle_gpu()

        # set the seed for reproducibility
        configure_seed(RANDOM_SEED)

        # define path json file containing the sensor columns
        numpy_columns_file = os.path.join(OUTPUT_FOLDER_PATH, SEGMENTED_DATA_FOLDER, SENSOR_COLS_JSON)

        # get sensor_columns
        sensor_columns = load_json_file(numpy_columns_file)[LOADED_SENSORS_KEY]

        # define path to save model
        project_path = os.path.dirname(os.path.abspath(__file__))
        model_save_path = create_dir(project_path,
                                     os.path.join("HAR", "dl",
                                                  f"trained_models_wsize-{window_size_samples}_seqlen-{seq_len}_batchsize-{batch_size}",
                                                  f"nm_{norm_type}", f"nt_{norm_method}",
                                                  "_".join(load_sensors)))

        print("training/testing model on generated dataset")
        train_dataloader, test_dataloader, test_dataloader_subject_wise, num_channels = (
            get_train_test_data(dataset_path, batch_size=batch_size,
                                load_sensors=load_sensors, sensor_columns=sensor_columns,
                                seq_len=seq_len, norm_method=norm_method, norm_type=norm_type,
                                balancing_type=balancing_type))

        # set model variables and parameters
        har_model = HARRnn(model_type=model_type, num_features=int(num_channels*seq_len), hidden_size=hidden_size, num_layers=num_layers,
                            num_classes=len(MAIN_ACTIVITY_LABELS), dropout=dropout)

        # generate model name
        model_name = f"{har_model.__class__.__name__}_{har_model.model_type}_hs-{har_model.hidden_size}_nl-{har_model.num_layers}_do-{int(har_model.dropout * 100)}"

        # put model on cuda device
        har_model.to(cuda_device)

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(har_model.parameters(), lr=lr)

        # run the training loop
        performance_history = run_model_training(model=har_model, model_save_path=model_save_path, model_name=model_name,
                                                 train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                                 test_dataloader_subject_wise=test_dataloader_subject_wise,
                                                 criterion=criterion, optimizer=optimizer, cuda_device=cuda_device,
                                                 num_epochs=num_epochs, patience=10)

        # plot the performance history
        plot_performance_history(performance_dict=performance_history, model_name=model_name, save_path=model_save_path)

        # save the model history as CSV
        performance_df = pd.DataFrame(performance_history)
        performance_df.to_csv(os.path.join(model_save_path, f"{model_name}_performance.csv"))

    if DL_POST_PROCESSING:

        # define sensors to use
        use_sensors = ['ACC', 'GYR']

        # set path to real-world dataset
        real_world_data_path = os.path.join(os.path.dirname(OUTPUT_FOLDER_PATH), "work_simulation", "raw_data")

        # define path to model state-dict
        # model_path = f"{DRIVE}:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\dl_results\\trained_models_wsize-500_seqlen-10_batchsize-64\\nm_global\\nt_z-score\\ACC_GYR\\HARRnn_lstm_hs-256_nl-1_do-30.pt"
        model_path = f"{DRIVE}:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\dl_results_new\\trained_models_wsize-250_seqlen-10_batchsize-64\\nm_subject\\nt_z-score\\ACC_GYR\\HARRnn_lstm_hs-256_nl-2_do-30.pt"

        # define path to statistics
        #stats_path = "F:\\Backup PrevOccupAI data\\Prevoccupai_HAR\\subject_data\\DL_Dataset\\w_500\\subject_statistics.json"
        stats_path = None

        # run post-processing optimization
        dl_optimize_post_processing(real_world_data_path, model_path, use_sensors, stats_path)







