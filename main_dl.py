
import os

# internal imports
from constants import SEGMENTED_DATA_FOLDER
from DL_HAR import generate_dataset


# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
GENERATE_DATASET = True



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
