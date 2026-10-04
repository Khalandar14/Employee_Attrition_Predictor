import streamlit as st
import pandas as pd
import joblib

# Loading the trained machine learning pipeline
model = joblib.load("employee_attrition_pipeline.pkl")

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("Employee Attrition Predictor")
st.write(
    "Enter the employee details below to predict whether the employee "
    "is likely to leave the organization."
)

st.subheader("Employee Information")

col1, col2 = st.columns(2)

with col1:
    business_travel = st.selectbox(
        "Business Travel",
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )

    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Other",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources"
        ]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

with col2:
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

st.subheader("Employee Metrics")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    daily_rate = st.number_input(
        "Daily Rate",
        min_value=102,
        max_value=1499,
        value=800
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=29,
        value=5
    )

    education = st.number_input(
        "Education",
        min_value=1,
        max_value=5,
        value=3
    )

    environment_satisfaction = st.number_input(
        "Environment Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=30,
        max_value=100,
        value=60
    )

    job_involvement = st.number_input(
        "Job Involvement",
        min_value=1,
        max_value=4,
        value=3
    )

    job_level = st.number_input(
        "Job Level",
        min_value=1,
        max_value=5,
        value=2
    )

    job_satisfaction = st.number_input(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=5000,
        step=500
    )

    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=0,
        value=14000,
        step=500
    )

    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=9,
        value=2
    )

with col2:
    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=11,
        max_value=25,
        value=15
    )

    performance_rating = st.number_input(
        "Performance Rating",
        min_value=3,
        max_value=4,
        value=3
    )

    relationship_satisfaction = st.number_input(
        "Relationship Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    stock_option_level = st.number_input(
        "Stock Option Level",
        min_value=0,
        max_value=3,
        value=1
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=8
    )

    training_times_last_year = st.number_input(
        "Training Times Last Year",
        min_value=0,
        max_value=6,
        value=3
    )

    work_life_balance = st.number_input(
        "Work Life Balance",
        min_value=1,
        max_value=4,
        value=3
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )

    years_in_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=18,
        value=3
    )

    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=2
    )

    years_with_curr_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=17,
        value=3
    )

input_data = pd.DataFrame({
    "Age": [age],
    "BusinessTravel": [business_travel],
    "DailyRate": [daily_rate],
    "Department": [department],
    "DistanceFromHome": [distance_from_home],
    "Education": [education],
    "EducationField": [education_field],
    "EmployeeCount": [1],
    "EmployeeNumber": [0],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "Gender": [gender],
    "HourlyRate": [hourly_rate],
    "JobInvolvement": [job_involvement],
    "JobLevel": [job_level],
    "JobRole": [job_role],
    "JobSatisfaction": [job_satisfaction],
    "MaritalStatus": [marital_status],
    "MonthlyIncome": [monthly_income],
    "MonthlyRate": [monthly_rate],
    "NumCompaniesWorked": [num_companies_worked],
    "Over18": ["Y"],
    "OverTime": [overtime],
    "PercentSalaryHike": [percent_salary_hike],
    "PerformanceRating": [performance_rating],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "StandardHours": [80],
    "StockOptionLevel": [stock_option_level],
    "TotalWorkingYears": [total_working_years],
    "TrainingTimesLastYear": [training_times_last_year],
    "WorkLifeBalance": [work_life_balance],
    "YearsAtCompany": [years_at_company],
    "YearsInCurrentRole": [years_in_current_role],
    "YearsSinceLastPromotion": [years_since_last_promotion],
    "YearsWithCurrManager": [years_with_curr_manager]
})

st.subheader("Prediction")

if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Attrition Probability: {probability:.2%}")

    if prediction == 1:
        st.error("⚠️ High Risk of Employee Attrition")
    else:
        st.success("✅ Low Risk of Employee Attrition")