from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from models.dataset_generator import get_training_features_and_labels

def evaluate_all_models():
    X, y = get_training_features_and_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr, Xte = scaler.fit_transform(X_train), scaler.transform(X_test)
    models = {"Logistic Regression": (LogisticRegression(max_iter=1000), Xtr, Xte), "Random Forest": (RandomForestClassifier(n_estimators=120, random_state=42), X_train, X_test), "ANN (MLP)": (MLPClassifier(hidden_layer_sizes=(32,16), max_iter=250, early_stopping=True, random_state=42), Xtr, Xte)}
    results = {}
    for name, (model, tr, te) in models.items():
        model.fit(tr, y_train)
        pred = model.predict(te)
        prob = model.predict_proba(te)[:, 1]
        results[name] = {"accuracy": float(accuracy_score(y_test, pred)), "auc": float(roc_auc_score(y_test, prob)), "confusion_matrix": confusion_matrix(y_test, pred).tolist(), "report": classification_report(y_test, pred, output_dict=True)}
    return results
