students = {
    "student1": {
        "Name": "Aqdas",
        "CGPA": 8.5,
        "Marks": 850,
        "Percentage": 85
    },
    "student2": {
        "Name": "Iqra Sadia",
        "CGPA": 9.0,
        "Marks": 900,
        "Percentage": 90
    },
    "student3": {
        "Name": "Abdullah",
        "CGPA": 7.8,
        "Marks": 780,
        "Percentage": 78
    },
    "student4": {
        "Name": "Faiza",
        "CGPA": 8.2,
        "Marks": 820,
        "Percentage": 82
    },
    "student5": {
        "Name": "Ali Usama",
        "CGPA": 8.9,
        "Marks": 890,
        "Percentage": 89
    }}
import numpy as np

# Extracting input features (marks and percentage) and target variable (CGPA)
X = np.array([[student["Marks"], student["Percentage"]] for student in students.values()])
y = np.array([student["CGPA"] for student in students.values()])
from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X, y)
from sklearn.metrics import mean_squared_error, r2_score

# Predict CGPA using the trained model
y_pred = model.predict(X)

# Calculate mean squared error
mse = mean_squared_error(y, y_pred)

# Calculate R-squared
r_squared = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
