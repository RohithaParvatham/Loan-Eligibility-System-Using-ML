import streamlit as st

from predict import get_category_options, load_model, predict_eligibility


st.set_page_config(page_title='Loan Eligibility Checker')
st.title('Loan Eligibility Checker')
st.caption('Enter applicant details to get an estimate from the trained model.')


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

category_options = get_category_options(model)

with st.form('applicant_details'):
    gender = st.selectbox('Gender', category_options['Gender'])
    married = st.selectbox('Married', category_options['Married'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    education = st.selectbox('Education', category_options['Education'])
    self_employed = st.selectbox('Self-employed', category_options['Self_Employed'])
    applicant_income = st.number_input('Applicant income', min_value=0, value=0, step=500)
    coapplicant_income = st.number_input(
        'Co-applicant income', min_value=0.0, value=0.0, step=500.0
    )
    loan_amount = st.number_input('Loan amount', min_value=0.0, value=0.0, step=10.0)
    loan_term = st.number_input('Loan term (months)', min_value=1, value=360, step=12)
    credit_history = st.selectbox('Credit history', [1, 0], format_func=lambda value: str(value))
    property_area = st.selectbox('Property area', category_options['Property_Area'])
    submitted = st.form_submit_button('Check eligibility')

if submitted:
    applicant = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
    }
    prediction = predict_eligibility(model, applicant)
    if prediction == 'Y':
        st.success('Model estimate: loan eligible')
    elif prediction == 'N':
        st.warning('Model estimate: loan not eligible')
    else:
        st.info(f'Model predicted loan status: {prediction}')

st.caption('This is an educational estimate, not a loan approval or financial advice.')
