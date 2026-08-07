# PrevOccupAI HAR

This project implements a Human Activity Recognition (HAR) pipeline for the PrevOccupAI dataset, built on inertial sensor data (accelerometer, gyroscope, magnetometer, and rotation vector) collected from smartphones. It covers the full workflow from raw sensor data to activity classification: raw signal pre-processing and segmentation, feature extraction, training and evaluation of both traditional machine learning (ML) and deep learning (DL) models to classify sitting, standing, and walking activities, and post-processing methods to refine model predictions on continuous, real-world recordings.

The codebase is organized into four main components, described below, and is run through two entry-point scripts: [main.py](main.py) for the ML pipeline and [main_dl.py](main_dl.py) for the DL pipeline.

## raw_data_processor

The [raw_data_processor](raw_data_processor) package turns raw smartphone sensor recordings into a clean, segmented dataset ready for feature extraction or DL dataset generation. It:

- Loads raw Android sensor files (ACC, GYR, MAG, ROT) from a recording ([load_sensor_data.py](raw_data_processor/load_sensor_data.py)) and interpolates them to a common, equidistant sampling rate ([interpolate.py](raw_data_processor/interpolate.py)).
- Filters and pre-processes the inertial signals, e.g., removing the gravitational component from the accelerometer and smoothing the rotation vector ([filters.py](raw_data_processor/filters.py), [pre_process.py](raw_data_processor/pre_process.py)).
- Detects and segments the individual (sub-)activities (e.g., `walking_slow`, `stairs_up`, `sitting`) performed under the acquisition protocol, based on the accelerometer signal ([segment_activities.py](raw_data_processor/segment_activities.py)).
- Saves each segmented (sub-)activity into its own file, together with a JSON header describing the sensor columns ([data_segmenter.py](raw_data_processor/data_segmenter.py)).

The entry point is `generate_segmented_dataset(...)`.

## feature_extractor

The [feature_extractor](feature_extractor) package converts the segmented sensor recordings into a windowed, feature-based dataset for the ML models. It:

- Slices the segmented signals into fixed-size, overlapping windows ([window.py](feature_extractor/window.py)).
- Extracts a broad set of time and frequency domain features per window using [TSFEL](https://github.com/fraunhoferportugal/tsfel), plus quaternion-derived features (e.g., geodesic distance between consecutive orientations) from the rotation vector sensor ([feature_extractor.py](feature_extractor/feature_extractor.py), [quaternion_features.py](feature_extractor/quaternion_features.py)).
- Labels each window with its main and sub-activity class and saves the resulting feature set per subject.

The entry point is `extract_features(...)`.

## Machine Learning (HAR/ml)

The [HAR/ml](HAR/ml) package trains and evaluates traditional ML classifiers (Random Forest, KNN, SVM) on the extracted features. It:

- Loads the per-subject feature files and balances the data per class, either on the main activities (sit/stand/walk) or their sub-classes ([load.py](HAR/ml/load.py), [HAR/data_balancer.py](HAR/data_balancer.py)).
- Applies model-agnostic feature selection (low-variance and highly-correlated feature removal, k-best selection) ([feature_selection.py](HAR/ml/feature_selection.py)).
- Selects the best-performing model via nested, group-aware cross-validation across Random Forest, KNN, and SVM ([cross_validation.py](HAR/ml/cross_validation.py), [model_selection.py](HAR/ml/model_selection.py)), using `Pipeline` objects throughout to avoid data leakage, and a pre-defined train/test split so models are only ever evaluated on unseen subjects.
- Trains and evaluates a final production model with tuned hyperparameters, both across all test subjects and per subject.

The relevant entry points are `perform_model_selection(...)` and `train_production_model(...)`.

## Deep Learning (HAR/dl)

The [HAR/dl](HAR/dl) package trains and evaluates DL models (LSTM, GRU, CNN-LSTM, 2D CNN-LSTM) directly on windowed, raw sensor sequences instead of hand-crafted features. It:

- Generates a windowed dataset from the segmented data for a given window size ([dataset_generator.py](HAR/dl/dataset_generator.py)), with subject- and window-level normalization and class balancing handled when the dataset is loaded.
- Defines the available model architectures ([models.py](HAR/dl/models.py)).
- Runs the training/test loop with early stopping, selects an idle GPU automatically, and plots/saves the resulting performance history ([train_test.py](HAR/dl/train_test.py), [utils.py](HAR/dl/utils.py)).

## Post-processing (HAR/post_process.py, HAR/post_processing_optimizer.py)

Both the ML and DL pipelines can be followed by post-processing of the raw, window-level predictions to produce smoother, more realistic activity sequences on continuous recordings. Implemented schemes include majority voting, threshold tuning between the "sitting" and "standing" classes, and heuristic minimum-duration correction of short, spurious segments ([HAR/post_process.py](HAR/post_process.py)). [HAR/post_processing_optimizer.py](HAR/post_processing_optimizer.py) runs a trained model on real-world data, applies these schemes, and can either evaluate a fixed set of parameters or optimize them (globally or per subject).

## main.py — running the machine learning pipeline

[main.py](main.py) drives the ML pipeline end to end. It is controlled by boolean flags at the top of the file rather than command-line arguments — set the flags for the stages you want to run and provide the relevant paths:

- `GENERATE_SEGMENTED_DATASET` — segments the raw recordings under `RAW_DATA_FOLDER_PATH` into `OUTPUT_FOLDER_PATH`.
- `EXTRACT_FEATURES` — extracts features from the segmented dataset.
- `ML_HAR` — enables the HAR training/evaluation block, further controlled by:
  - `ML_MODEL_SELECTION` — runs nested cross-validation to compare Random Forest, KNN, and SVM.
  - `ML_TRAIN_PRODUCTION_MODEL` — trains and evaluates the final production model.
  - `ML_POST_PROCESSING` — applies post-processing to a trained model's predictions on real-world data (`OPTIMIZE_INDIVIDUAL_PARAM` switches between per-subject parameter optimization and evaluation with a fixed, pre-identified parameter set).

Adjust `RAW_DATA_FOLDER_PATH`, `OUTPUT_FOLDER_PATH`, and (for post-processing) `me_dataset_path` to point at your local data, then run:

```bash
python main.py
```

## main_dl.py — running the deep learning pipeline

[main_dl.py](main_dl.py) drives the DL pipeline and is configured through command-line arguments (see `parser` in the script for the full list, including dataset parameters such as `--window_size_s`, `--seq_len`, `--load_sensors`, `--norm_method`, and model parameters such as `--model_type`, `--num_epochs`, `--batch_size`, `--hidden_size`, `--dropout`). Three stages are toggled via boolean constants at the top of the file:

- `GENERATE_DATASET` — builds the windowed DL dataset from the segmented data if it doesn't already exist for the requested window size.
- `TRAIN_TEST_MODEL` — trains and evaluates the selected model architecture, saving the trained weights, a performance plot, and a CSV of the training history under `HAR/dl/`.
- `DL_POST_PROCESSING` — runs post-processing optimization for a trained DL model on real-world data.

Example run, training a 2D CNN-LSTM model:

```bash
python main_dl.py --model_type cnnlstm2d --window_size_s 5 --seq_len 25 --load_sensors ACC GYR --num_epochs 40 --batch_size 64
```

## Installation

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The deep learning pipeline additionally requires `torch` and `gputil`, which are not listed in `requirements.txt`.
