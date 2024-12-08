"""
evaluate_model.py
-------------------
This script evaluates the trained models using metrics such as accuracy, 
precision, recall, and F1-score. It also generates evaluation reports 
and confusion matrices.

"""

from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
