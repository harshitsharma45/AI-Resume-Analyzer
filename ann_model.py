import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from models.dataset_generator import get_training_features_and_labels
class ANNModelManager:
    """Lightweight ANN using sklearn MLP; avoids TensorFlow deployment overhead."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPClassifier(hidden_layer_sizes=(32, 16, 8), early_stopping=True, max_iter=250, random_state=42)
        self.is_trained = False
        self.history = {}
    def load_or_train(self):
        if not self.is_trained:
            X, y = get_training_features_and_labels()
            self.train(X, y)
        return self
    def train(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        self.is_trained = True
        self.history = {"loss": list(self.model.loss_curve_)}
    def predict(self, feature_vector):
        if not self.is_trained: self.load_or_train()
        X = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        prob = float(self.model.predict_proba(self.scaler.transform(X))[0, 1])
        return {"prediction": "Shortlist" if prob >= .5 else "Not Shortlisted", "probability": round(prob * 100, 1), "is_shortlisted": prob >= .5}
