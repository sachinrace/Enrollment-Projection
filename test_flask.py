

from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)

# Load the CSV data
df = pd.read_csv(r'C:\Users\sparmar\OneDrive - Northwestern Polytechnic\Documents\Experiment\Programming\Python\RockPaperScissor-Random\enrollment.csv')

# Preprocess the data and train the model
def train_model():
    # Assuming 'Year' and 'Total_Enrollment' are the features we're using for prediction
    features = ['Year', 'IRCC_Reforms', 'Economic_Indicators', 'Public_Perception']
    target = 'Total_Enrollment'

    X = df[features]
    y = df[target]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    accuracy = 100 - (rmse / np.mean(y_test)) * 100

    return model, accuracy

# Train the model and get the accuracy
model, accuracy = train_model()

# Flask Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    # Predict future enrollment for the next 5 years (e.g., from 2024 to 2029)
    future_years = pd.DataFrame({
        'Year': [2024, 2025, 2026, 2027, 2028],
        'IRCC_Reforms': [1, 0, 0, 1, 0],  # Assuming IRCC reforms remain active
        'Economic_Indicators': [0.1, 0.8, 0.3, 0.8, 0.1],  # Example economic indicators
        'Public_Perception': [0.3, 0.6, 0.9, 0.1, 0.6]  # Example public perception scores
    })

    # Get predictions
    future_enrollments = model.predict(future_years)

    # Convert the data into JSON format for the frontend
    predictions = {
        'years': future_years['Year'].tolist(),
        'enrollments': future_enrollments.tolist(),
        'accuracy': accuracy
    }

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)