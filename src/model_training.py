import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import joblib

def load_data():
    df = pd.read_csv("C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\final_model_data.csv")
    X = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log(model, param_grid, model_name):
    mlflow.set_experiment("credit_scoring_experiment")
    with mlflow.start_run(run_name=model_name):
        X_train, X_test, y_train, y_test = load_data()
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        preds = best_model.predict(X_test)
        proba = best_model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, preds, output_dict=True)
        roc_auc = roc_auc_score(y_test, proba)

        # Log metrics
        mlflow.log_metric("accuracy", report['accuracy'])
        mlflow.log_metric("precision", report['1']['precision'])
        mlflow.log_metric("recall", report['1']['recall'])
        mlflow.log_metric("f1_score", report['1']['f1-score'])
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        mlflow.sklearn.log_model(best_model, model_name)
        joblib.dump(best_model, f"models/{model_name}.pkl")

        print(f"{model_name} done. Best params: {grid_search.best_params_}")

if __name__ == "__main__":
    # Logistic Regression
    lr_params = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }
    train_and_log(LogisticRegression(max_iter=1000), lr_params, "logistic_regression")

    # Random Forest
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }
    train_and_log(RandomForestClassifier(random_state=42), rf_params, "random_forest")
