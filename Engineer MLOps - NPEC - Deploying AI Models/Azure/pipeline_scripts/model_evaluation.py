import os
import sys
from typing import Generator, Tuple

import numpy as np
from sklearn.metrics import f1_score, jaccard_score
from tensorflow.keras.models import Model
import tensorflow as tf

try:
    from image_data_generator import test_generator
except ImportError:
    from .image_data_generator import test_generator

def convert_masks(
    predictions: np.ndarray, masks: np.ndarray, binary: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts prediction probabilities to binary or categorical labels.

    Args:
        predictions (np.ndarray): Array of predicted probabilities.
        masks (np.ndarray): Array of true label masks.
        binary (bool): Specifies if the task is binary classification (True) or multiclass classification (False).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of predicted labels and true labels, adjusted for binary or categorical evaluation.

    Author: Benjamin Graziadei
    """
    if binary:
        return (predictions > 0.5).astype(np.int32), masks
    else:
        return np.argmax(predictions, axis=-1), np.argmax(masks, axis=-1)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, binary: bool) -> dict:
    """
    Calculates and returns F1-score and Jaccard Index (IoU) for the predicted results against true labels.

    Args:
        y_true (np.ndarray): True label masks.
        y_pred (np.ndarray): Predicted label masks.
        binary (bool): If True, calculates binary metrics. Otherwise, calculates weighted metrics for multiclass.

    Returns:
        dict: Dictionary containing 'f1_score' and 'iou_score' for each class or overall if binary.

    Author: Benjamin Graziadei
    """
    metrics = {}

    def calculate_f1_iou(tp, fp, fn):
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        iou_score = tp / (tp + fp + fn + 1e-7)
        return f1_score.numpy(), iou_score.numpy()

    if not binary:
        y_true = tf.argmax(y_true, axis=-1)  # Assuming y_true is one-hot encoded
        y_pred = tf.argmax(y_pred, axis=-1)  # Converting probabilities to class labels

        class_f1_scores = []
        class_iou_scores = []

        for i in range(y_true.shape[-1]):
            y_pred_i = tf.cast(y_pred == i, tf.float32)
            y_true_i = tf.cast(y_true == i, tf.float32)

            tp = tf.reduce_sum(y_pred_i * y_true_i)
            fp = tf.reduce_sum(y_pred_i * (1 - y_true_i))
            fn = tf.reduce_sum((1 - y_pred_i) * y_true_i)
            intersection = tp
            union = tp + fp + fn

            f1_score, iou_score = calculate_f1_iou(tp, fp, fn)
            class_f1_scores.append(f1_score)
            class_iou_scores.append(iou_score)

        metrics['f1_score'] = class_f1_scores
        metrics['iou_score'] = class_iou_scores
    else:
        # Binary metric calculation with flattening
        y_true_flat = tf.reshape(y_true, (-1,))  # Flatten ground truth
        y_pred_flat = tf.reshape(y_pred, (-1,))  # Flatten predictions

        # Thresholding for binary prediction using sigmoid activation (assuming)
        y_pred_binary = tf.cast(y_pred_flat > 0.5, tf.float32)  # Threshold at 0.5
        y_true_flat = tf.cast(y_true_flat, tf.float32)

        tp = tf.reduce_sum(y_pred_binary * y_true_flat)
        fp = tf.reduce_sum(y_pred_binary * (1 - y_true_flat))
        fn = tf.reduce_sum((1 - y_pred_binary) * y_true_flat)
        intersection = tp
        union = tp + fp + fn

        f1_score, iou_score = calculate_f1_iou(tp, fp, fn)
        metrics['f1_score'] = f1_score
        metrics['iou_score'] = iou_score

    return metrics


def evaluate_model(model: Model, generator: Generator) -> None:
    """
    Evaluates a trained model using test data from a generator and prints the performance metrics.

    Args:
        model_path (str): Path to the trained model to be evaluated.
        generator (Generator): Generator that provides batches of test images and corresponding masks.

    Outputs:
        Print the calculated performance metrics for each batch of test data.

    Author: Benjamin Graziadei
    """
    num_classes = model.layers[-1].output_shape[-1]
    binary = num_classes == 1

    all_metrics = {}

    for images, name, masks  in generator:
        preds = model.predict(images)
        preds, masks = convert_masks(preds, masks, binary)
        metrics = calculate_metrics(masks, preds, binary)

        # Accumulate metrics
        for key, value in metrics.items():
            if key in all_metrics:
                all_metrics[key].append(value)
            else:
                all_metrics[key] = [value]

    final_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    print(
        f"Evaluation Metrics for {'Binary' if binary else 'Multiclass'} classification:"
    )
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    return final_metrics


def main(model: Model, test_path) -> None:
    test_gen = test_generator(8, test_path)
    return evaluate_model(model, test_gen)