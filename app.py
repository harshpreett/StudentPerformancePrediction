import os
import json
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the saved models 
model = joblib.load('models/model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

DEFAULT_VALUES = {
    'school': 'GP', 'sex': 'F', 'age': 16, 'address': 'U', 'famsize': 'GT3',
    'Pstatus': 'T', 'Medu': 3, 'Fedu': 3, 'Mjob': 'other', 'Fjob': 'other',
    'reason': 'course', 'guardian': 'mother', 'traveltime': 1, 'studytime': 2,
    'failures': 0, 'schoolsup': 'no', 'famsup': 'no', 'paid': 'no',
    'activities': 'no', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
    'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 3, 'Dalc': 1,
    'Walc': 1, 'health': 3, 'absences': 0
}

def get_fi_data():
    if os.path.exists("static/fi.json"):
        with open("static/fi.json", "r") as f:
            data = json.load(f)
            return list(data.keys()), list(data.values())
    return [], []

@app.route('/')
def home():
    fi_labels, fi_values = get_fi_data()
    return render_template('index.html', prediction=None, fi_labels=fi_labels, fi_values=fi_values)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = DEFAULT_VALUES.copy()
    form_data = request.form.to_dict()
    
    numeric_cols = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 'absences', 'health']
    for col in numeric_cols:
        if col in form_data and form_data[col]:
            user_input[col] = int(form_data[col])
            
    cat_cols = ['sex', 'higher', 'internet', 'famsup', 'activities']
    for col in cat_cols:
        if col in form_data and form_data[col]:
            user_input[col] = form_data[col]
            
    df = pd.DataFrame([user_input])
    X_processed = preprocessor.transform(df)
    
    prediction = model.predict(X_processed)[0]
    proba = model.predict_proba(X_processed)[0][1]
    
    result = "PASS" if prediction == 1 else "FAIL"
    confidence = proba if prediction == 1 else (1 - proba)
    
    fi_labels, fi_values = get_fi_data()
    
    return render_template('index.html', 
                           prediction=result, 
                           confidence=f"{confidence * 100:.1f}%",
                           fi_labels=fi_labels,
                           fi_values=fi_values)

if __name__ == '__main__':
    app.run(debug=True)