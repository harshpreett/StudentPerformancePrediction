import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the saved models from Day 1
model = joblib.load('models/model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Baseline defaults for the dataset so we don't need a 30-field HTML form
DEFAULT_VALUES = {
    'school': 'GP', 'sex': 'F', 'age': 16, 'address': 'U', 'famsize': 'GT3',
    'Pstatus': 'T', 'Medu': 3, 'Fedu': 3, 'Mjob': 'other', 'Fjob': 'other',
    'reason': 'course', 'guardian': 'mother', 'traveltime': 1, 'studytime': 2,
    'failures': 0, 'schoolsup': 'no', 'famsup': 'no', 'paid': 'no',
    'activities': 'no', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
    'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 3, 'Dalc': 1,
    'Walc': 1, 'health': 3, 'absences': 0, 'G1': 10, 'G2': 10
}

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Merge user form data with defaults
    user_input = DEFAULT_VALUES.copy()
    form_data = request.form.to_dict()
    
    # Update and convert numeric fields
    for col in ['age', 'studytime', 'failures', 'absences', 'G1', 'G2']:
        if col in form_data and form_data[col]:
            user_input[col] = int(form_data[col])
            
    # Process and Predict
    df = pd.DataFrame([user_input])
    X_processed = preprocessor.transform(df)
    
    prediction = model.predict(X_processed)[0]
    proba = model.predict_proba(X_processed)[0][1]
    
    result = "PASS" if prediction == 1 else "FAIL"
    confidence = proba if prediction == 1 else (1 - proba)
    
    return render_template('index.html', 
                           prediction=result, 
                           confidence=f"{confidence * 100:.1f}%")

if __name__ == '__main__':
    app.run(debug=True)