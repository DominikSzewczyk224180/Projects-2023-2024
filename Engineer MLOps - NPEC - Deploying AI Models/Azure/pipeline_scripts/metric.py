import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric


class F1IoUMetric(Metric):
    """
    Custom metric class to calculate F1 score and Intersection over Union (IoU) for segmentation tasks.

    This metric can be used for both binary and multiclass segmentation models. It computes individual class-wise
    F1 scores and IoUs, and provides averaged metrics as well.

    Attributes:
    num_classes (int): The number of classes in the segmentation task.
    name (str): Name of the metric.
    is_binary (bool): True if the task is binary segmentation, else False.
    tp (list of tf.Variable): True positives for each class.
    fp (list of tf.Variable): False positives for each class.
    fn (list of tf.Variable): False negatives for each class.
    intersection (list of tf.Variable): Intersection values for IoU calculation for each class.
    union (list of tf.Variable): Union values for IoU calculation for each class.
    """

    def __init__(self, num_classes: int, name: str = "f1_iou_metric", **kwargs):
        super(F1IoUMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.is_binary = num_classes == 1
        self.tp = [
            self.add_weight(name=f"true_positives_{i}", initializer="zeros", shape=())
            for i in range(num_classes)
        ]
        self.fp = [
            self.add_weight(name=f"false_positives_{i}", initializer="zeros", shape=())
            for i in range(num_classes)
        ]
        self.fn = [
            self.add_weight(name=f"false_negatives_{i}", initializer="zeros", shape=())
            for i in range(num_classes)
        ]
        self.intersection = [
            self.add_weight(name=f"intersection_{i}", initializer="zeros", shape=())
            for i in range(num_classes)
        ]
        self.union = [
            self.add_weight(name=f"union_{i}", initializer="zeros", shape=())
            for i in range(num_classes)
        ]
        print(f"F1IoUMetric initialized with {num_classes} classes")

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None
    ) -> None:
        """
        Updates the state of the metric based on the input values.

        Args:
        y_true (tf.Tensor): True labels for the data.
        y_pred (tf.Tensor): Predicted results by the model.
        sample_weight (tf.Tensor): Optional sample weights (not used in this implementation).
        """
        if not self.is_binary:
            y_true = tf.argmax(y_true, axis=-1)  # Assuming y_true is one-hot encoded
            y_pred = tf.argmax(
                y_pred, axis=-1
            )  # Converting probabilities to class labels

            for i in range(self.num_classes):
                y_pred_i = tf.cast(y_pred == i, tf.float32)
                y_true_i = tf.cast(y_true == i, tf.float32)

                tp = tf.reduce_sum(y_pred_i * y_true_i)
                fp = tf.reduce_sum(y_pred_i * (1 - y_true_i))
                fn = tf.reduce_sum((1 - y_pred_i) * y_true_i)
                intersection = tp
                union = tp + fp + fn

                self.tp[i].assign_add(tp)
                self.fp[i].assign_add(fp)
                self.fn[i].assign_add(fn)
                self.intersection[i].assign_add(intersection)
                self.union[i].assign_add(union)
        else:
            # Binary metric calculation with flattening
            y_true_flat = tf.reshape(y_true, (-1,))  # Flatten ground truth
            y_pred_flat = tf.reshape(y_pred, (-1,))  # Flatten predictions

            # Thresholding for binary prediction using sigmoid activation (assuming)
            y_pred_binary = tf.cast(y_pred_flat > 0.5, tf.float32)  # Threshold at 0.5
            y_true_flat = tf.cast(y_true_flat, tf.float32)

            intersection = tp = tf.reduce_sum(y_pred_binary * y_true_flat)
            fp = tf.reduce_sum(y_pred_binary * (1 - y_true_flat))
            fn = tf.reduce_sum((1 - y_pred_binary) * y_true_flat)
            union = tp + fp + fn

            self.tp[0].assign_add(tp)
            self.fp[0].assign_add(fp)
            self.fn[0].assign_add(fn)
            self.intersection[0].assign_add(intersection)
            self.union[0].assign_add(union)

    def result(self) -> dict:
        """
        Calculates and returns the metric results as a dictionary.

        Returns:
        dict: A dictionary containing the overall F1 score and IoU, along with individual class scores.
        """
        precision = [
            tp / (tp + fp + tf.keras.backend.epsilon())
            for tp, fp in zip(self.tp, self.fp)
        ]
        recall = [
            tp / (tp + fn + tf.keras.backend.epsilon())
            for tp, fn in zip(self.tp, self.fn)
        ]
        f1_scores = [
            2 * (p * r) / (p + r + tf.keras.backend.epsilon())
            for p, r in zip(precision, recall)
        ]
        ious = [
            intsec / (uni + tf.keras.backend.epsilon())
            for intsec, uni in zip(self.intersection, self.union)
        ]

        results = {
            "f1_score": tf.reduce_mean(tf.stack(f1_scores)),
            "iou": tf.reduce_mean(tf.stack(ious)),
        }
        if not self.is_binary:
            for i in range(self.num_classes):
                results[f"f1_c{i}"] = f1_scores[i]
                results[f"iou_c{i}"] = ious[i]

        return results

    def reset_state(self) -> None:
        """
        Resets all the state variables of the metric (i.e., true positives, false positives, etc.).
        """
        for i in range(self.num_classes):
            self.tp[i].assign(0)
            self.fp[i].assign(0)
            self.fn[i].assign(0)
            self.intersection[i].assign(0)
            self.union[i].assign(0)


def f1(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the F1 score, the harmonic mean of precision and recall.

    Args:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: F1 score.
    """

    def recall_m(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1_score


def iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the Intersection over Union (IoU) metric.

    Args:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: IoU score.
    """

    def f(y_true, y_pred):
        y_pred = K.cast(y_pred > 0.5, dtype=K.floatx())
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
        union = total - intersection
        iou = (intersection + K.epsilon()) / (union + K.epsilon())
        return iou

    return K.mean(f(y_true, y_pred), axis=0)
