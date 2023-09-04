from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model_filename = 'xgboost_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    location = int(request.form['location'])
    monthly_bill = float(request.form['monthly_bill'])
    total_usage_gb = int(request.form['total_usage_gb'])
    age_group = int(request.form['age_group'])
    subscription_duration = int(request.form['subscription_duration'])
    total_money_spent = float(request.form['total_money_spent'])

    data = pd.DataFrame({
        'Gender': [gender],
        'Location': [location],
        'Monthly_Bill': [monthly_bill],
        'Total_Usage_GB': [total_usage_gb],
        'Age_Group': [age_group],
        'Subscription_Duration': [subscription_duration],
        'Total_Money_Spent': [total_money_spent]
    })

    data_1d = data.values.reshape(1, -1)
    data_scaled = scaler.transform(data_1d)
    prediction = model.predict(data_scaled)

    if prediction == 0:
        result = 'Not Churn'
    else:
        result = 'Churn'

    return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
