import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.dataset_generator import get_training_features_and_labels
class MLModelManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
        self.is_trained = False
    def load_or_train(self):
        if not self.is_trained:
            X, y = get_training_features_and_labels()
            self.train_models(X, y)
        return self
    def train_models(self, X, y):
        X = np.asarray(X, dtype=float)
        self.lr_model.fit(self.scaler.fit_transform(X), y)
        self.rf_model.fit(X, y)
        self.is_trained = True
    def predict(self, feature_vector):
        if not self.is_trained: self.load_or_train()
        X = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        lr = float(self.lr_model.predict_proba(self.scaler.transform(X))[0, 1])
        rf = float(self.rf_model.predict_proba(X)[0, 1])
        return {"logistic_regression": {"prediction": "Shortlist" if lr >= .5 else "Not Shortlisted", "probability": round(lr * 100, 1)}, "random_forest": {"prediction": "Shortlist" if rf >= .5 else "Not Shortlisted", "probability": round(rf * 100, 1)}}
