import unittest

from predict import predict_eligibility


class RecordingModel:
    def predict(self, applicant_data):
        self.applicant_data = applicant_data
        return ['Y']


class PredictEligibilityTests(unittest.TestCase):
    def test_three_plus_dependents_are_converted_for_model(self):
        model = RecordingModel()
        applicant = {
            'Gender': 'Male',
            'Married': 'Yes',
            'Dependents': '3+',
            'Education': 'Graduate',
            'Self_Employed': 'No',
            'ApplicantIncome': 5000,
            'CoapplicantIncome': 1000.0,
            'LoanAmount': 150.0,
            'Loan_Amount_Term': 360,
            'Credit_History': '1',
            'Property_Area': 'Urban',
        }

        result = predict_eligibility(model, applicant)

        self.assertEqual(result, 'Y')
        self.assertEqual(model.applicant_data.loc[0, 'Dependents'], 3)
        self.assertEqual(model.applicant_data.loc[0, 'Credit_History'], 1)
        self.assertEqual(applicant['Dependents'], '3+')

    def test_numeric_dependent_count_is_converted_to_integer(self):
        model = RecordingModel()
        applicant = {
            'Gender': 'Female',
            'Married': 'No',
            'Dependents': '2',
            'Education': 'Graduate',
            'Self_Employed': 'Yes',
            'ApplicantIncome': 4000,
            'CoapplicantIncome': 0.0,
            'LoanAmount': 100.0,
            'Loan_Amount_Term': 360,
            'Credit_History': 0,
            'Property_Area': 'Rural',
        }

        predict_eligibility(model, applicant)

        self.assertEqual(model.applicant_data.loc[0, 'Dependents'], 2)


if __name__ == '__main__':
    unittest.main()
