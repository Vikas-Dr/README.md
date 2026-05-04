from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("models/iris_rf.joblib")


def train_model(model_path: Path = MODEL_PATH) -> float:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "target_names": iris.target_names,
            "feature_names": iris.feature_names,
            "accuracy": accuracy,
        },
        model_path,
    )

    return accuracy


if __name__ == "__main__":
    score = train_model()
    print(f"Saved model to {MODEL_PATH} with accuracy={score:.4f}")
