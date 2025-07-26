# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import joblib

# Step 1: Load the dataset
data = pd.read_csv('Loan_Data.csv')  # Replace 'loan_dataset.csv' with your dataset file path

# Display the first few rows of the dataset
print("Initial Data Preview:")
print(data.head())

# Step 2: Data Preprocessing

# Drop irrelevant columns like 'Loan_ID' if they are not useful for prediction
if 'Loan_ID' in data.columns:
    data = data.drop('Loan_ID', axis=1)

# Handling missing values
imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

for col in categorical_cols:
    if data[col].dtype == 'object':  # Only encode if the column is still an object
        data[col] = label_encoder.fit_transform(data[col])

# Handle 'Dependents' column specifically
if 'Dependents' in data.columns:
    # Replace '3+' with a numeric value (e.g., 3 or 4)
    data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)

# Ensure target column is also encoded if it's categorical
if data['Loan_Status'].dtype == 'object':
    data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])

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
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Confusion Matrix Plot

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Eligible', 'Eligible'], yticklabels=['Not Eligible', 'Eligible'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 8: Feature Importance Plot

# Feature importance
feature_importances = model.feature_importances_

# Create a DataFrame for feature importance
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
plt.title('Feature Importance Plot')
plt.show()

# Step 9: ROC Curve Plot

# Calculate probabilities and ROC curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Step 10: Save the Model (Optional)

# Save the trained model to a file
joblib.dump(model, 'loan_eligibility_model.pkl')
print("\nModel saved as 'loan_eligibility_model.pkl'.")
