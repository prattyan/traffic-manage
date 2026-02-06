"""
Script to create and train a simple LSTM model for traffic prediction.
This generates the traffic_lstm.h5 file needed by main.py
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)

# Create synthetic traffic data for training
# Simulate vehicle counts over time
def generate_traffic_data(samples=1000, sequence_length=10):
    """Generate synthetic traffic data."""
    X = []
    y = []
    
    for _ in range(samples):
        # Generate a sequence of vehicle counts (0-100)
        sequence = np.random.randint(5, 100, sequence_length)
        X.append(sequence.reshape(-1, 1))
        
        # Next vehicle count is influenced by the trend
        next_count = min(100, max(0, int(np.mean(sequence)) + np.random.randint(-10, 10)))
        y.append(next_count)
    
    return np.array(X), np.array(y)

# Generate training data
print("Generating synthetic traffic data...")
X_train, y_train = generate_traffic_data(samples=1000, sequence_length=10)

# Normalize the data
X_train = X_train / 100.0  # Normalize to 0-1
y_train = y_train / 100.0  # Normalize to 0-1

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

# Build LSTM model
print("Building LSTM model...")
model = Sequential([
    LSTM(64, activation='relu', input_shape=(10, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output between 0-1 for normalized vehicle counts
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
print("Training LSTM model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Save the model
print("Saving model as traffic_lstm.h5...")
model.save('traffic_lstm.h5')
print("✓ traffic_lstm.h5 created successfully!")

# Display model summary
print("\nModel Summary:")
model.summary()
