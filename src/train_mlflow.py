import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

os.makedirs("models", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Digits-Classification-MLOps")

# Load processed CSV if available; otherwise load & scale sklearn digits directly
csv_path = "data/processed/digits_processed.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    X = df[[c for c in df.columns if c.startswith("pixel_")]]
    y = df["target"]
else:
    digits = load_digits(as_frame=True)
    X = pd.DataFrame(
        StandardScaler().fit_transform(digits.data),
        columns=digits.feature_names,
    )
    y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM_RBF": SVC(C=1.0, kernel="rbf", random_state=42),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        joblib.dump(model, f"models/{name}.joblib")
        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")