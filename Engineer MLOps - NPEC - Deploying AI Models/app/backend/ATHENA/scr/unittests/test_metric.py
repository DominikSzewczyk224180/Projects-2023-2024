import os
import sys
import unittest
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metric import f1, iou, F1IoUMetric

class TestF1IoUMetrics(unittest.TestCase):
    def setUp(self):
        self.metric = F1IoUMetric(num_classes=1)

    def test_f1(self):
        y_true = tf.constant([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0.8, 0.2, 0.6, 0.1], [0.4, 0.9, 0.3, 0.2]], dtype=tf.float32)

        # Compute the F1 score using the function
        f1_score = f1(y_true, y_pred).numpy()

        # Expected F1 score calculated manually
        expected_f1_score = 0.8571428  # Adjust this value based on the actual manual calculation

        self.assertAlmostEqual(f1_score, expected_f1_score, places=4)

    def test_iou(self):
        y_true = tf.constant(np.random.randint(0, 2, (1, 10, 10, 1)), dtype=tf.float32)
        y_pred = tf.constant(np.random.rand(1, 10, 10, 1), dtype=tf.float32)

        # Helper function to calculate intersection, union, and IoU
        def calculate_iou(y_true, y_pred):
            y_pred = K.cast(y_pred > 0.5, dtype=K.floatx())
            intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
            total = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
            union = total - intersection
            iou = (intersection + K.epsilon()) / (union + K.epsilon())
            return iou.numpy()

        # Compute the IoU using the function
        iou_score = iou(y_true, y_pred).numpy()

        # Expected IoU score calculated manually or using another verified method
        expected_iou_score = calculate_iou(y_true, y_pred)

        # Check the IoU score against the expected value element-wise
        np.testing.assert_almost_equal(iou_score, expected_iou_score, decimal=4)

    def test_binary_segmentation(self) -> None:
        """
        Test the binary segmentation metric calculation to ensure it handles simple cases correctly.

        Author: Benjamin Graziadei
        """
        metric = F1IoUMetric(num_classes=1)
        y_pred = tf.constant([0.1, 0.4, 0.6, 0.8, 0.99], shape=(5, 1, 1, 1))
        y_true = tf.constant([0, 0, 1, 1, 1], shape=(5, 1, 1, 1))
        
        metric.update_state(y_true, y_pred)
        result = metric.result()

        self.assertAlmostEqual(result['f1_score'].numpy(), 1, places=5)
        self.assertAlmostEqual(result['iou'].numpy(), 1, places=3)

    def test_multiclass_segmentation(self) -> None:
        """
        Test multiclass segmentation metric to ensure it handles diverse scenarios across multiple classes.

        Author: Benjamin Graziadei
        """
        metric = F1IoUMetric(num_classes=3)
        y_pred = tf.constant([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], shape=(5, 1, 1, 3))
        y_true = tf.constant([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ], shape=(5, 1, 1, 3))

        metric.update_state(y_true, y_pred)
        result = metric.result()

        self.assertAlmostEqual(result['f1_c0'].numpy(), 0.5, places=5)
        self.assertAlmostEqual(result['iou_c0'].numpy(), 0.333, places=3)
        self.assertAlmostEqual(result['f1_c1'].numpy(), 0.5, places=5)
        self.assertAlmostEqual(result['iou_c1'].numpy(), 0.333, places=3)
        self.assertAlmostEqual(result['f1_c2'].numpy(), 1.0, places=5)
        self.assertAlmostEqual(result['iou_c2'].numpy(), 1.0, places=5)

    def test_metric_reset(self) -> None:
        """
        Test that the metric reset functionality correctly resets all metric state variables to zero.

        Author: Benjamin Graziadei
        """
        metric = F1IoUMetric(num_classes=2)
        
        # Simulate updates
        y_pred = tf.random.uniform((10, 1, 1, 2), minval=0, maxval=1)
        y_true = tf.random.uniform((10, 1, 1, 2), minval=0, maxval=1, dtype=tf.int32)
        metric.update_state(y_true, y_pred)
        
        # Reset the metric
        metric.reset_state() 
        
        # Check reset state
        for i in range(2):
            self.assertEqual(K.get_value(metric.tp[i]), 0)
            self.assertEqual(K.get_value(metric.fp[i]), 0)
            self.assertEqual(K.get_value(metric.fn[i]), 0)
            self.assertEqual(K.get_value(metric.intersection[i]), 0)
            self.assertEqual(K.get_value(metric.union[i]), 0)

    def test_update_state(self):
        y_true = tf.constant([[0, 1, 1, 0, 1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.9, 0.8, 0.2, 0.7]], dtype=tf.float32)

        self.metric.update_state(y_true, y_pred)

        tp = K.get_value(self.metric.tp[0])
        fp = K.get_value(self.metric.fp[0])
        fn = K.get_value(self.metric.fn[0])
        intersection = K.get_value(self.metric.intersection[0])
        union = K.get_value(self.metric.union[0])

        self.assertEqual(tp, 3.0)
        self.assertEqual(fp, 0.0)
        self.assertEqual(fn, 0.0)
        self.assertEqual(intersection, 3.0)
        self.assertEqual(union, 3.0)

    def test_result(self):
        y_true = tf.constant([[0, 1, 1, 0, 1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.9, 0.8, 0.2, 0.7]], dtype=tf.float32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        expected_f1_score = 1.0
        expected_iou = 1.0

        self.assertAlmostEqual(result["f1_score"].numpy(), expected_f1_score, places=5)
        self.assertAlmostEqual(result["iou"].numpy(), expected_iou, places=5)



    
    
if __name__ == '__main__':
    unittest.main()
