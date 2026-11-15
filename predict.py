from pathlib import Path

import joblib
import pandas as pd


categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']


def load_model():
    model_path = Path(__file__).resolve().with_name('loan_eligibility_model.pkl')
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run source.py first to train and save it."
        )

    model = joblib.load(model_path)
    if not hasattr(model, 'named_steps') or 'preprocessor' not in model.named_steps:
        raise ValueError(
            "The saved model is outdated. Run source.py once to create the preprocessing pipeline."
        )
    return model


def get_category_options(model):
    categorical_encoder = (
        model.named_steps['preprocessor']
        .named_transformers_['categorical']
        .named_steps['encoder']
    )
    return {
        col: [str(value) for value in options]
        for col, options in zip(categorical_cols, categorical_encoder.categories_)
    }


def predict_eligibility(model, applicant):
    prepared_applicant = dict(applicant)
    dependents = str(prepared_applicant['Dependents']).strip()
    prepared_applicant['Dependents'] = 3 if dependents == '3+' else int(dependents)
    prepared_applicant['Credit_History'] = int(prepared_applicant['Credit_History'])
    return model.predict(pd.DataFrame([prepared_applicant]))[0]


def prompt_choice(field, choices):
    choices_text = ', '.join(choices)
    while True:
        value = input(f"{field} ({choices_text}): ").strip()
        match = next((choice for choice in choices if choice.lower() == value.lower()), None)
        if match is not None:
            return match
        print(f"Please enter one of: {choices_text}")


def prompt_number(field, allow_decimal=False):
    while True:
        value = input(f"{field}: ").strip()
        try:
            return float(value) if allow_decimal else int(value)
        except ValueError:
            expected = 'a number' if allow_decimal else 'a whole number'
            print(f"Please enter {expected}.")


def main():
    model = load_model()
    category_options = get_category_options(model)

    print("Enter applicant details to check loan eligibility.")
    applicant = {
        'Gender': prompt_choice('Gender', category_options['Gender']),
        'Married': prompt_choice('Married', category_options['Married']),
        'Dependents': prompt_choice('Dependents', ['0', '1', '2', '3+']),
        'Education': prompt_choice('Education', category_options['Education']),
        'Self_Employed': prompt_choice('Self_Employed', category_options['Self_Employed']),
        'ApplicantIncome': prompt_number('ApplicantIncome'),
        'CoapplicantIncome': prompt_number('CoapplicantIncome', allow_decimal=True),
        'LoanAmount': prompt_number('LoanAmount'),
        'Loan_Amount_Term': prompt_number('Loan_Amount_Term'),
        'Credit_History': int(prompt_choice('Credit_History', ['0', '1'])),
        'Property_Area': prompt_choice('Property_Area', category_options['Property_Area']),
    }
    prediction = predict_eligibility(model, applicant)
    if prediction == 'Y':
        print("\nResult: Loan eligible")
    elif prediction == 'N':
        print("\nResult: Loan not eligible")
    else:
        print(f"\nPredicted loan status: {prediction}")


if __name__ == '__main__':
    main()
