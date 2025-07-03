import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import joblib
import os

# Create models folder
os.makedirs("models", exist_ok=True)

# Load and split data
def load_data():
    df = pd.read_csv("C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\final_model_data.csv")
    X = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train, evaluate, and log a model
def train_and_log(model, param_grid, model_name):
    X_train, X_test, y_train, y_test = load_data()
    mlflow.set_experiment("credit_scoring_experiment")

    with mlflow.start_run(run_name=model_name):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc
        })

        # Log model
        mlflow.sklearn.log_model(best_model, model_name)
        joblib.dump(best_model, f"models/{model_name}.pkl")

        print(f"{model_name} done. Best params: {grid_search.best_params_}")
        print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc:.4f}")

        return best_model, f1  # return F1 for comparison

if __name__ == "__main__":
    # Logistic Regression
    lr_params = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }
    lr_model, lr_f1 = train_and_log(LogisticRegression(max_iter=1000), lr_params, "logistic_regression")

    # Random Forest
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }
    rf_model, rf_f1 = train_and_log(RandomForestClassifier(random_state=42), rf_params, "random_forest")

    # Choose and register best model
    best_model = rf_model if rf_f1 > lr_f1 else lr_model
    best_model_name = "random_forest" if rf_f1 > lr_f1 else "logistic_regression"
    print(f"\n Best model is: {best_model_name} with F1 score: {max(rf_f1, lr_f1):.4f}")

    # Register model
    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/{best_model_name}",
        name="BestCreditScoringModel"
    )
