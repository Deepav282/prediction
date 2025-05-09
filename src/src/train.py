import numpy as np
from preprocess import load_and_preprocess, create_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Parameters
WINDOW_SIZE = 60
EPOCHS = 50
BATCH_SIZE = 32

# Load data
data, scaler = load_and_preprocess('../data/your_stock_data.csv')

# Prepare sequences
X, y = create_sequences(data, WINDOW_SIZE)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

model.save('stock_lstm_model.h5')
