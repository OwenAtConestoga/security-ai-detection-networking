import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Example preprocessing steps
    data = data.dropna()
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Labels
    return train_test_split(X, y, test_size=0.2, random_state=42)
