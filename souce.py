# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Step 1: Load the dataset
data = pd.read_csv("C:\Users\Admin\Downloads\Loan-Eligibility-Prediction-main\Loan-Eligibility-Prediction-main\project ml\Loan_Data.csv")  # Replace 'loan_dataset.csv' with your dataset file path

# Display the first few rows of the dataset
print("Initial Data Preview:")
print(data.head())

# Step 2: Data Preprocessing

# Handling missing values
imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Display the first few rows after preprocessing
print("\nData After Preprocessing:")
print(data.head())

# Step 3: Split the Data into Training and Test Sets

# Define the feature variables (X) and the target variable (y)
X = data.drop('Loan_Status', axis=1)  # Drop the target column
y = data['Loan_Status']  # Target column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Step 5: Make Predictions

# Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the Model (Optional)

# Save the trained model to a file
joblib.dump(model, 'loan_eligibility_model.pkl')
print("\nModel saved as 'loan_eligibility_model.pkl'.")
