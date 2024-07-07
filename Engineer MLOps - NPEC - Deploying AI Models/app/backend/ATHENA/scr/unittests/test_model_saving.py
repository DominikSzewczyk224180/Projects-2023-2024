import os
import shutil
import unittest
import numpy as np
import json
import tensorflow as tf

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from model_saving import (
    save_model,
    get_layer_description,
    create_model_description,
    create_full_model_architecture,
    training_configuration,
    save_selected_info,
)

class TestModelUtils(unittest.TestCase):
    def setUp(self):
        self.model = Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(10, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.folder_path = "test_model_dir"
        self.model_name = "test_model"
        self.validation_data = {"accuracy": 0.85}  # Placeholder for actual validation metrics
        
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
    
    def tearDown(self):
        if os.path.exists(self.folder_path):
            shutil.rmtree(self.folder_path)
    
    def test_save_model(self):
        file_path = save_model(self.model, self.folder_path, self.model_name)
        self.assertTrue(os.path.exists(file_path))
    
    def test_get_layer_description(self):
        descriptions = [
            "Convolutional layer: 32 filters, kernel size (3, 3), activation function 'relu'",
            "MaxPooling layer: pool size (2, 2)",
            "Flatten layer: flattens the input",
            "Dense layer: 128 units, activation function 'relu'",
            "Dense layer: 10 units, activation function 'softmax'",
        ]
        for layer, desc in zip(self.model.layers, descriptions):
            self.assertEqual(get_layer_description(layer), desc)
    
    def test_create_model_description(self):
        description = create_model_description(self.model)
        self.assertIn("Model Description:", description)
        self.assertIn("Convolutional layer: 32 filters", description)
    
    def test_create_full_model_architecture(self):
        architecture_json = create_full_model_architecture(self.model)
        self.assertIn('"class_name": "Sequential"', architecture_json)
    
    def test_training_configuration(self):
        config = training_configuration(self.model, self.validation_data)
        self.assertIn("Adam", config)
        self.assertIn("Loss Function: sparse_categorical_crossentropy", config)
        self.assertIn("accuracy: 0.8500", config)
    
    def test_save_selected_info(self):
        save_selected_info(
            self.model, self.folder_path, [1, 2, 3, 4], self.validation_data, self.model_name
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.folder_path, f"{self.model_name}.h5")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.folder_path, f"{self.model_name}_info.txt")
            )
        )

if __name__ == "__main__":
    unittest.main()
