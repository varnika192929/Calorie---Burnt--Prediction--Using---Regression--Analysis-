import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Example dataset
data = {
    'Age': [25, 30, 35, 40, 45],
    'Weight': [70, 80, 75, 65, 72],
    'Height': [175, 168, 180, 165, 170],
    'Gender': ['M', 'F', 'M', 'M', 'F'],  # This will need to be encoded
    'Activity_Level': [1.2, 1.375, 1.55, 1.725, 1.9],  # Example: Sedentary, Lightly active, etc.
    'Calories_Burnt': [2500, 2200, 2700, 2300, 2400]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical data (Gender in this case)
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

# Split data into features and target
X = df[['Age', 'Weight', 'Height', 'Gender', 'Activity_Level']]
y = df['Calories_Burnt']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Take user input
age = float(input("Enter age: "))
weight = float(input("Enter weight (kg): "))
height = float(input("Enter height (cm): "))
gender = input("Enter gender (M/F): ")
activity_level = float(input("Enter activity level (e.g., 1.2, 1.375, 1.55, 1.725, 1.9): "))

# Encode gender
gender_encoded = 0 if gender.upper() == 'M' else 1

# Create new data point
new_data = pd.DataFrame([[age, weight, height, gender_encoded, activity_level]], 
                        columns=['Age', 'Weight', 'Height', 'Gender', 'Activity_Level'])

# Scale the new data
new_data_scaled = scaler.transform(new_data)

# Predict calories burnt
predicted_calories = model.predict(new_data_scaled)
print(f"Predicted Calories Burnt: {predicted_calories[0]}")