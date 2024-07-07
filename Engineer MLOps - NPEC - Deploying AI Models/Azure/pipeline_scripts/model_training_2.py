import argparse
import mlflow
import tensorflow as tf
import os
import sys
import keras_tuner as kt
import numpy as np
import logging
import warnings
from data_loading import load_data_uri, load_data, extract_shape
from logging_metrics import plotting_metrics, TrainingLogger
from metric import f1, iou
from model_architectures import build_unet_model
from model_hyperparameter_tuning import MyTuner, train_and_evaluate
from model_saving import save_model

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages are logged (default), 1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed
# set tensorflow memory growth to true

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Suppress all warnings
warnings.filterwarnings("ignore")

def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a model with optional hyperparameter tuning.')

    parser.add_argument("--use_uri", action='store_true',
                        help="Use the dataset from data asets.")
    parser.add_argument('--train_image_path', type=str,
                        help='Train image path for the model.')
    parser.add_argument('--train_mask_path', type=str,
                        help='Train masks path for the model.')
    parser.add_argument('--val_image_path', type=str,
                        help='Validation images path for the model.')
    parser.add_argument('--val_mask_path', type=str,
                        help='Validation masks path for the model.')
    parser.add_argument('--model_save_path', type=str, default="../../models",
                        help='Path where to save the model.')
    parser.add_argument('--model_name', type=str,
                        help='The name of the model')
    parser.add_argument('--early_stopping', type=str, choices=['yes', 'no'], default='no',
                        help='Whether to use early stopping (yes/no)')
    parser.add_argument('--n_epochs', type=int, default=15,
                        help='Number of epochs to train the model.')
    parser.add_argument('--hyperparameter_tuning', type=str, choices=['yes', 'no'], default='no',
                        help='Whether to perform hyperparameter tuning (yes/no).')
    parser.add_argument('--n_trials', type=int, default=5,
                        help='Number of trials for hyperparameter tuning.')
    parser.add_argument('--save_model', type=str, choices=['yes', 'no'], default='yes',
                        help='Whether to save the model (yes/no).')

    args = parser.parse_args()

    # Set up logging
    if not args.use_uri:
        logging.basicConfig(filename=f'log/{args.model_name}.log', filemode= 'w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Start MLflow tracking
    mlflow.start_run()
    mlflow.tensorflow.autolog()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available")
    else:
        print("GPU is not available")

    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"Gpus: {physical_devices}")
    if len(physical_devices) > 0:
        logging.info("Setting memory growth for GPU")
    else:
        logging.info("No GPU found :(")

    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        logging.info("Could not set memory growth for GPU")
        pass

    patch_size=256
    channels=1

    if not args.use_uri:
        # Create data generators
        train, val, test = load_data()
        # patch_size, channels = extract_shape(train)
    else:
        train = load_data_uri(args.train_image_path, args.train_mask_path)
        val = load_data_uri(args.val_image_path, args.val_mask_path)

    logging.info("Data for training and validation was loaded")

    # Create the model
    model = build_unet_model(patch_size, patch_size, channels)

    # Log the model summary to MLflow

    with open("model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    mlflow.log_artifact("model_summary.txt")

    cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, mode='min')

    if args.hyperparameter_tuning.lower() == 'yes':
        logging.info("Hyperparameter tuning will be performed!")
        tuner = MyTuner(
            model=model,
            train_data=train,
            valid_data=val,
            epochs=args.n_epochs,
            early_stopping=cb if args.early_stopping.lower() == 'yes' else None,
            metrics=['accuracy', f1, iou],
            objective='val_accuracy',
            max_trials=args.n_trials,
            overwrite=True,
            directory=None,
            project_name=None
        )

        tuner.search()

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        logging.info("Hyperparameter tuning completed!")
        logging.info("Train a model with the hyperparameters")

        model, history, evaluation = train_and_evaluate(
                        model,
                        train,
                        val,
                        args.n_epochs,
                        cb if args.early_stopping.lower() == 'yes' else None,
                        best_hp.get('batch_size'),
                        best_hp.get("learning_rate"),
                        best_hp.get("optimizer"),
                        metrics=['accuracy', f1, iou])

        plotting_metrics(history)

    else:
        logging.info("Hyperparameter tuning will not be performed")
        model, history, evaluation = train_and_evaluate(
                        model, 
                        train, 
                        val, 
                        args.n_epochs, 
                        [cb] if args.early_stopping.lower() == 'yes' else None, 
                        metrics=['accuracy', f1, iou])
        
        print(evaluation)
        
        plotting_metrics(history)

    if args.save_model.lower() == 'yes':
        save_model(model, args.model_name, args.model_save_path)
    else:
        logging.info("Model saving skipped")

    mlflow.end_run()


if __name__ == '__main__':

    main()
