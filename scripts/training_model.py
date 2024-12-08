"""
train_model.py
----------------
This script trains machine learning models using the preprocessed datasets. 
The models include supervised (e.g., Random Forest, SVM) and unsupervised 
algorithms (e.g., K-means, Isolation Forest).

"""

from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    dump(model, '../models/random_forest.pkl')
    return model
