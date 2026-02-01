import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

X_train = np.random.rand(100, 1, 1)
y_train = np.random.rand(100, 1)

model = Sequential([
    LSTM(16, activation='relu', input_shape=(1, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("Generating model weights...")
model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

model.save("traffic_lstm.h5")
print("✅ Success! 'traffic_lstm.h5' has been generated.")