# 🎓 Student Performance Prediction Dashboard

An end-to-end Machine Learning project that predicts whether a student will **Pass** or **Fail** using the UCI Student Performance Dataset. The project includes data preprocessing, model comparison, hyperparameter tuning, and deployment using Flask.

## 🚀 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- HTML/CSS
- Bootstrap
- Chart.js
- Joblib

---

## 📊 Dataset

- **Source:** UCI Student Performance Dataset
- **Students:** 395
- **Features:** 30+
- **Target:** Pass / Fail
- **Target Creation:** `Pass = 1 if G3 >= 10 else 0`

> G1 and G2 were excluded from training to predict performance using only student background and behavioral features.

---

## ⚙️ Machine Learning Pipeline

- Data Loading
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- One-Hot Encoding
- Feature Scaling
- Train/Test Split
- Model Training
- Model Comparison
- Hyperparameter Tuning (GridSearchCV)
- Model Saving
- Flask Deployment

---

## 🤖 Models Evaluated

| Model | Accuracy |
|--------|---------:|
| Logistic Regression | **87.34%** |
| Decision Tree | **86.08%** |
| Random Forest | **87.34%** |
| Tuned XGBoost | **87.34%** |

### Best Model Performance

| Metric | Score |
|--------|------:|
| Accuracy | **87.34%** |
| Precision | **0.96** |
| Recall | **0.85** |
| F1-Score | **0.90** |

---

## 📁 Project Structure

```
StudentPerformancePrediction/
│
├── app.py
├── train.py
├── utils.py
├── requirements.txt
│
├── dataset/
├── models/
├── notebooks/
├── templates/
├── static/
└── README.md
```

---

## ▶️ Installation

```bash
git clone https://github.com/harshpreett/StudentPerformancePrediction.git

cd StudentPerformancePrediction

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 📌 Features

- Automated preprocessing pipeline
- Multiple ML model comparison
- Hyperparameter tuning
- Model persistence using Joblib
- Interactive Flask web application
- Student Pass/Fail prediction

---

## 👨‍💻 Author

**Harshpreet Singh**

B.Tech CSE (Data Science)

Punjab Engineering College (PEC)

GitHub: https://github.com/harshpreett
