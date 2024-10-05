import joblib
import numpy as np
from sklearn.linear_model import LinearRegression 

# Simulating a model for demonstration purposes
# In your case, use your trained model from the earlier code
X = np.array([[60], [65], [70], [75], [80]])  # Heights
Y = np.array([120, 130, 140, 150, 160])  # Weights

# Initialize and train the model
model = LinearRegression()
model.fit(X, Y)

# Save the model to a file
joblib.dump(model, 'linear_regression_model.pkl')

print("Model saved successfully.")
