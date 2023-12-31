from flask import Flask, render_template, request,Response
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

scaler = pickle.load(open('Model/standardScaler.pkl', 'rb'))
model = pickle.load(open('Model/predictionModel.pkl', 'rb'))


@app.route('/',  methods = ['GET','POST'])
def index():
    result = ''

    if request.method == 'POST':
        try:
            # Get values from the form using request.form
            Pregnancies = int(request.form['pregnancies'])
            Glucose = float(request.form['glucose'])
            BloodPressure = float(request.form['bloodPressure'])
            SkinThickness = float(request.form['skinThickness'])
            Insulin = float(request.form['insulin'])
            BMI = float(request.form['bmi'])
            DiabetesPedigreeFunction = float(request.form['diabetesPedigreeFunction'])
            Age = float(request.form['age'])
            features = [Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin, BMI, DiabetesPedigreeFunction, Age]
            new_scaled_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin, BMI, DiabetesPedigreeFunction, Age]])
            result = model.predict(new_scaled_data)

            return render_template('result.html', result = result, feature =features)
        except ValueError:
            return render_template('index.html', error = True)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
