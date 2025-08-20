
import os
from sklearn.model_selection import GroupShuffleSplit


# internal imports
from constants import SEGMENTED_DATA_FOLDER, CLASS_INSTANCES_JSON, RANDOM_SEED
from HAR.dl import generate_dataset, HARDataset
from HAR.dl import DL_DATASET
from file_utils import load_json_file

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

        generate_dataset(segmented_data_path, OUTPUT_FOLDER_PATH, window_size=5, default_input_file_type='.npy')

    if TRAIN_TEST_MODEL:

        window_size_samples = 500

        print("training/testing model on generated dataset")

        # get train/test subject split
        # load the subject ids
        subject_IDs = list(load_json_file(os.path.join(OUTPUT_FOLDER_PATH, DL_DATASET, f'w_{window_size_samples}', CLASS_INSTANCES_JSON)).keys())

        # init splitter
        splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_SEED)
        train_idx, test_idx = next(splitter.split(subject_IDs, groups=subject_IDs))

        # split the data
        subject_IDs_train = [subject_IDs[idx] for idx in train_idx]
        subject_IDs_test = [subject_IDs[idx] for idx in test_idx]

        # load data
        train_dataset = HARDataset(data_path=os.path.join(OUTPUT_FOLDER_PATH, DL_DATASET, f'w_{window_size_samples}'),
                                   subject_ids=subject_IDs_train,
                                   norm_method="z-score", norm_type="global", balancing_type=None)

        test_dataset = HARDataset(data_path=os.path.join(OUTPUT_FOLDER_PATH, DL_DATASET, f'w_{window_size_samples}'),
                                   subject_ids=subject_IDs_test,
                                   norm_method="z-score", norm_type="global", balancing_type=None)

        print(f"total samples train: {len(train_dataset)}")
        print(f"total samples test: {len(test_dataset)}")

        # X, y_main, y_sub = dataset[0]
        #
        # print("Sample shape:", X.shape)
        # print("Main activity label:", y_main)
        # print("Sub activity label:", y_sub)
        #
        # X, y_main, y_sub = dataset[232]
        #
        # print("Sample shape:", X.shape)
        # print("Main activity label:", y_main)
        # print("Sub activity label:", y_sub)

        # train model

        # test model
