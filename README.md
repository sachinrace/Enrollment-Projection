# Enrollment Projection

# International Student Enrollment Prediction

This project is a web application built using Flask that predicts future international student enrollment based on several factors, including IRCC reforms, economic indicators, and public perception. The application utilizes a Random Forest regression model trained on historical enrollment data to make predictions for the next five years.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)


## Technologies Used

- Python
- Flask
- Pandas
- scikit-learn (for machine learning)
- HTML/CSS (for frontend rendering)

## Features

- Predicts international student enrollment for the next five years.
- Displays predictions in a user-friendly web interface.
- Provides accuracy metrics for the model's predictions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/enrollment-projection.git
   cd enrollment-projection
2. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the required dependencies:
   pip install -r requirements.txt
4. Ensure you have the enrollment.csv dataset in the appropriate directory.

## Usage
1. Start the Flask application:
   python app.py
2. Open your web browser and navigate to http://127.0.0.1:5000/ to access the application.
3. Visit the /predict endpoint to view the predicted enrollments for the next five years in JSON format.

## How It Works
Data Loading: The application reads the historical enrollment data from a CSV file.

Model Training: The Random Forest regression model is trained using the features:

Year
IRCC_Reforms
Economic_Indicators
Public_Perception
The model is evaluated using Mean Squared Error (MSE) to calculate accuracy.

Prediction: When a user accesses the /predict route, the application generates future enrollment predictions based on predefined values for the next five years.
