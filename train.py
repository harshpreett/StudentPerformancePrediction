import os
import zipfile
import urllib.request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

def main():
    # --- 1. Load Dataset (with auto-download fallback) ---
    DATASET_PATH = 'dataset/student-mat.csv'
    if not os.path.exists(DATASET_PATH):
        os.makedirs('dataset', exist_ok=True)
        print("Downloading dataset from UCI repository...")
        url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
        zip_path = "dataset/student.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract('student-mat.csv', 'dataset/')
        os.remove(zip_path)

    df = pd.read_csv(DATASET_PATH, sep=';') 

    # --- 2. Create Pass Target and Drop G3 ---
    df['Pass'] = (df['G3'] >= 10).astype(int)
    X = df.drop(columns=['G3', 'Pass'])
    y = df['Pass']

    # --- 3. Build Preprocessing Pipeline ---
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # --- 4. Train/Test Split & Preprocessing ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save the preprocessor
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Preprocessor saved to models/preprocessor.pkl")

    # --- 5. Base Model Training & Evaluation ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    print("\n" + "-"*65)
    print(f"{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1':<5}")
    print("-" * 65)

    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"{name:<22} | {acc:.4f}   | {prec:.4f}    | {rec:.4f} | {f1:.4f}")

    print("-" * 65 + "\n")

    # --- 6. Hyperparameter Tuning for XGBoost ---
    print("Starting GridSearchCV for XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    grid_search.fit(X_train_processed, y_train)
    
    best_xgb = grid_search.best_estimator_
    print(f"Best XGBoost Parameters: {grid_search.best_params_}")
    
    # Evaluate best model
    best_y_pred = best_xgb.predict(X_test_processed)
    best_acc = accuracy_score(y_test, best_y_pred)
    print(f"Tuned XGBoost Accuracy: {best_acc:.4f}")

    # --- 7. Save Best Model ---
    joblib.dump(best_xgb, 'models/model.pkl')
    print("Tuned XGBoost model saved to models/model.pkl")
    print("\nDay 1 Deliverables Complete!")

if __name__ == "__main__":
    main()