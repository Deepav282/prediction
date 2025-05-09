import numpy as np
import matplotlib.pyplot as plt

predicted = np.load('predicted.npy')
real = np.load('real.npy')

plt.figure(figsize=(12,6))
plt.plot(real, color='blue', label='Actual Price')
plt.plot(predicted, color='red', label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
