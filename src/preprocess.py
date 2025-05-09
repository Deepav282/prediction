import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(csv_path, feature_col='Close'):
    df = pd.read_csv(csv_path)
    df = df[[feature_col]].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)
    return scaled, scaler

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
