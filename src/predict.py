import numpy as np
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess, create_sequences

WINDOW_SIZE = 60

data, scaler = load_and_preprocess('../data/your_stock_data.csv')
X, y = create_sequences(data, WINDOW_SIZE)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = load_model('stock_lstm_model.h5')
predictions = model.predict(X)
predicted_prices = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y.reshape(-1, 1))

np.save('predicted.npy', predicted_prices)
np.save('real.npy', real_prices)
