from flask import Flask, render_template, request
import numpy as np
import joblib  # Import joblib to load the saved model

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (ensure 'linear_regression_model.pkl' is in the same directory)
model = joblib.load('linear_regression_model.pkl')

# Home page - render input form
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html exists in 'templates' folder

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get height input from form
    height = float(request.form['height'])

    # Reshape input for the model (since the model expects a 2D array)
    height_reshaped = np.array([[height]])

    # Make prediction using the loaded model
    prediction = model.predict(height_reshaped)

    # Render the result on the HTML page
    return render_template('index.html', prediction_text=f'Predicted Weight: {prediction[0]:.2f}')


if __name__ == "__main__":
    app.run(debug=True)


