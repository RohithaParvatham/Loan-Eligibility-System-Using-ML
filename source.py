# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
import joblib

# Step 1: Load the dataset
project_dir = Path(__file__).resolve().parent
data = pd.read_csv(project_dir / 'Loan_Data.csv')

# Display the first few rows of the dataset
print("Initial Data Preview:")
print(data.head())

# Step 2: Prepare features and target
if 'Loan_ID' in data.columns:
    data = data.drop('Loan_ID', axis=1)

categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
numeric_cols = [
    'Dependents',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
]
data['Dependents'] = pd.to_numeric(data['Dependents'].replace('3+', 3), errors='coerce')
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status'].astype(str)
# Step 3: Split the data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_cols),
    ('categorical', categorical_pipeline, categorical_cols),
])
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
])
baseline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000)),
])

# Step 4: Compare models using cross-validation on the training set
cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'f1_macro': 'f1_macro',
}
cross_validation_results = []
model_candidates = {
    'Random Forest': model,
    'Logistic Regression': baseline_model,
}
cross_validation_scores = {}
for name, candidate in model_candidates.items():
    scores = cross_validate(
        candidate,
        X_train,
        y_train,
        cv=cross_validation,
        scoring=scoring,
    )
    cross_validation_scores[name] = scores
    cross_validation_results.append({
        'Model': name,
        'Accuracy (mean +/- SD)': (
            f"{scores['test_accuracy'].mean():.3f} +/- "
            f"{scores['test_accuracy'].std():.3f}"
        ),
        'Balanced accuracy (mean +/- SD)': (
            f"{scores['test_balanced_accuracy'].mean():.3f} +/- "
            f"{scores['test_balanced_accuracy'].std():.3f}"
        ),
        'Macro F1 (mean +/- SD)': (
            f"{scores['test_f1_macro'].mean():.3f} +/- "
            f"{scores['test_f1_macro'].std():.3f}"
        ),
    })

print("\n5-Fold Cross-Validation on Training Data:")
print(pd.DataFrame(cross_validation_results).set_index('Model').to_string())

selected_model_name = max(
    cross_validation_scores,
    key=lambda name: cross_validation_scores[name]['test_f1_macro'].mean(),
)
model = model_candidates[selected_model_name]
print(f"\nSelected model by mean cross-validated macro F1: {selected_model_name}")

# Evaluate the selected model once on the held-out test set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print(
    "Classification Report:\n",
    classification_report(
        y_test,
        y_pred,
        labels=['N', 'Y'],
        target_names=['Not Eligible', 'Eligible'],
        zero_division=0,
    ),
)

conf_matrix = confusion_matrix(y_test, y_pred, labels=['N', 'Y'])

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Eligible', 'Eligible'], yticklabels=['Not Eligible', 'Eligible'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

feature_names = model.named_steps['preprocessor'].get_feature_names_out()
classifier = model.named_steps['classifier']
if hasattr(classifier, 'feature_importances_'):
    feature_importances = classifier.feature_importances_
    feature_plot_title = 'Feature Importance'
else:
    feature_importances = abs(classifier.coef_[0])
    feature_plot_title = 'Absolute Logistic Regression Coefficients'
features_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
plt.title(feature_plot_title)
plt.show()

positive_class_index = list(model.classes_).index('Y')
y_pred_prob = model.predict_proba(X_test)[:, positive_class_index]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label='Y')
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

joblib.dump(model, project_dir / 'loan_eligibility_model.pkl')
print("\nModel pipeline saved as 'loan_eligibility_model.pkl'.")
