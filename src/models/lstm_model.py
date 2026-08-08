import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm(input_shape):
    model = models.Sequential([
        layers.LSTM(64, input_shape=input_shape),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

