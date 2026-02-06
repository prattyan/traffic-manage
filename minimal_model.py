"""
Simple script to create a minimal LSTM model and save it.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("Creating minimal LSTM model...")

# Create a simple LSTM model
model = Sequential([
    LSTM(32, input_shape=(10, 1)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

print("Saving model...")
model.save('traffic_lstm.h5')
print("✓ traffic_lstm.h5 created successfully!")
