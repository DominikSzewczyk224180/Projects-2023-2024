import argparse
import subprocess
import os
import sys
import pkg_resources

try:
    from . import landmark_detection, predictions
    from . import model_evaluation, model_saving, model_training
except ImportError:
    import landmark_detection, predictions
    import model_evaluation, model_saving, model_training

root_dir = pkg_resources.resource_filename(__name__, '')

def main() -> None:
    """
    Main function to process data, train a model, and make predictions using the machine learning pipeline.

    Args:
        None

    Returns:
        None

    Author: Benjamin Graziadei
    """
    parser = argparse.ArgumentParser(description="Process, train, and predict using the machine learning pipeline.")
    
    parser.add_argument('--process', action='store_true', help="Process new data")
    parser.add_argument('--process_mode', type=str, choices=['custom', 'default'], help="Processing mode (custom or default)")
    parser.add_argument('--dir', type=str, help="Directory to process")

    parser.add_argument('--train', action='store_true', help="Train a new model")
    parser.add_argument('--train_path', type=str, help="Path to training data")
    parser.add_argument('--depth', type=int, choices=[0, 1], help="Depth of the model (required if --train is specified and must be 0 or 1)")
    parser.add_argument('--predict_path', type=str, help="Path for prediction")
    parser.add_argument('--model_name', type=str, help="Name of the model to save")

    args = parser.parse_args()
    
    if args.process:
        process_args = ["python", "scr/process_data.py"]
        if args.process_mode:
            process_args.extend(["--mode", args.process_mode])
        if args.dir:
            process_args.extend(["--dir", args.dir])
        subprocess.run(process_args)

    if args.train:
        if args.depth is None or args.train_path is None or args.model_name is None:
            parser.error("--train requires --depth, --train_path, and --model_name to be specified")
        if args.depth not in [0, 1]:
            parser.error("--depth must be 0 or 1")
        model, history = model_training.train(args.depth, args.train_path)
        metrics = model_evaluation.main(model, args.train_path)
        model_saving.model_saving(model, metrics, args.model_name)

    if args.predict_path:
        path = predictions.main(args.model_name, args.predict_path)
        print(f"Predictions saved to {path}")
        landmark_detection.Landmark_detecion(path)

if __name__ == "__main__":
    main()