import tensorflow as tf
from tensorflow.python.keras import layers, models
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_images(df, img_size=(224, 224)):
    """Load and preprocess images"""
    images = []
    for img_path in df["image_path"]:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        images.append(img / 255.0)
    return np.array(images)


def build_simple_cnn(input_shape=(224, 224, 3)):
    """Simple CNN architecture"""
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),  # Regression output
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model():
    # Load data
    df = pd.read_csv("data/processed/regression_data.csv")
    images = load_images(df)
    targets = df["defect_percent"].values

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        images, targets, test_size=0.2, random_state=42
    )

    # Build model
    model = build_simple_cnn()

    # Train
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32
    )

    # Save model
    model.save("models/simple_cnn_regression.h5")
    return history


if __name__ == "__main__":
    train_model()
