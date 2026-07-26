import os
import zipfile
import urllib.request
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

def main():
    # 1. Dataset Loading (with auto-download fallback)
    DATASET_PATH = 'dataset/student-mat.csv'
    if not os.path.exists(DATASET_PATH):
        os.makedirs('dataset', exist_ok=True)
        url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
        zip_path = "dataset/student.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract('student-mat.csv', 'dataset/')
        os.remove(zip_path)

    df = pd.read_csv(DATASET_PATH, sep=";")
    df["Pass"] = (df["G3"] >= 10).astype(int)
    
    # Drop G1, G2, G3, and Pass
    X = df.drop(columns=["G1", "G2", "G3", "Pass"])
    y = df["Pass"]

    # 2. Pipeline setup
    cat = X.select_dtypes(include=["object", "bool"]).columns
    num = X.select_dtypes(include=["int64", "float64"]).columns

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore"))
        ]), cat)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)

    # 3. Model Training
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
    }

    best_model = None
    best_name = ""
    best_acc = 0

    print("-" * 70)
    print(f'{"Model":25} {"Accuracy":>10}')
    print("-" * 70)

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"{name:25} {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    # 4. XGBoost Tuning
    params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5]
    }

    print("\nTuning XGBoost...")
    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="logloss"),
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=8,
        verbose=0
    )
    grid.fit(X_train, y_train)
    
    tuned = grid.best_estimator_
    tuned_acc = accuracy_score(y_test, tuned.predict(X_test))
    
    print("\nBest XGBoost Parameters:")
    print(grid.best_params_)
    print(f"Tuned XGBoost Accuracy: {tuned_acc:.4f}")

    if tuned_acc > best_acc:
        best_model = tuned
        best_name = "Tuned XGBoost"
        best_acc = tuned_acc

    # 5. Save Models
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(pre, "models/preprocessor.pkl")

    # 6. Save Dashboard Assets (Confusion Matrix & Feature Importance)
    os.makedirs("static", exist_ok=True)
    
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("static/cm.png")
    plt.close()

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = best_model.coef_[0]
        
    feature_names = pre.get_feature_names_out()
    fi_dict = {name.split("__")[-1]: float(imp) for name, imp in zip(feature_names, importances)}
    sorted_fi = dict(sorted(fi_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:8])
    
    with open("static/fi.json", "w") as f:
        json.dump(sorted_fi, f)

    print(f"\nBest Model Saved: {best_name}")
    print(f"Final Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()