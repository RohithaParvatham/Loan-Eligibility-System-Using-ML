# Loan Eligibility Demo

This project trains a loan-status classifier and provides command-line and Streamlit prediction interfaces. It is an educational example, not a real loan approval system.

## Setup

From the project folder:

```powershell
python -m pip install -r requirements.txt
```

Keep `Loan_Data.csv` in the project folder.

## Train and evaluate

```powershell
python source.py
```

Training uses a stratified 80/20 train/test split. Missing-value handling, scaling, and categorical encoding are part of the model pipeline so they are fitted only on training data.

Before the final holdout evaluation, the script compares Random Forest with Logistic Regression using five-fold stratified cross-validation on the training portion. It reports mean and standard deviation for accuracy, balanced accuracy, and macro F1. The held-out test portion is used once for the final classification report, balanced accuracy, confusion matrix, and ROC curve.

The model with the highest mean cross-validated macro F1 is selected and saved. Review the cross-validation table and the held-out class-wise recall before describing the results; accuracy alone can hide missed ineligible applications.

The training run displays the confusion matrix, feature importance, and ROC curve. Close each plot window to continue; the model is saved after the plots.

## Get predictions

Run the command-line prompts:

```powershell
python predict.py
```

Or launch the browser form:

```powershell
python -m streamlit run app.py
```

Train the model first so `loan_eligibility_model.pkl` exists. The web app normally opens at `http://localhost:8501`.

## Tests

```powershell
python -m unittest discover -v
```

The tests cover applicant-field normalization, including the dataset's `3+` dependents value.

## Limitations

The reported scores describe this dataset and split only; they do not establish performance on future applicants. The data may be small, outdated, or unrepresentative. Evaluate fairness and data quality before any real-world use, and do not treat model output as financial advice or an approval decision.
