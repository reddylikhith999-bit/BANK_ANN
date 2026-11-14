# ============================================================
# BANK MARKETING PREDICTION — STREAMLIT APP (ANN)
# ============================================================

import os
import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Bank Term Deposit Prediction (ANN)",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# 1. LOAD ANN MODEL FROM LOCAL PROJECT FOLDER
# ============================================================

MODEL_FILE = "model_ann.pkl"

@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"{MODEL_FILE} not found in project folder")

    return joblib.load(MODEL_FILE)


try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading ANN model: {e}")
    st.stop()


# ============================================================
# 2. USER INTERFACE
# ============================================================

st.title("🧠 Bank Term Deposit Prediction (ANN Model)")

st.markdown(
    "Enter customer details below to predict term deposit subscription "
    "using an Artificial Neural Network."
)

st.sidebar.header("Customer Input Form")

age = st.sidebar.slider("Age", 18, 95, 35)

job = st.sidebar.selectbox(
    "Job Type",
    ["admin.", "blue-collar", "technician", "services",
     "management", "retired", "entrepreneur", "self-employed",
     "housemaid", "unemployed", "student", "unknown"]
)

marital = st.sidebar.selectbox(
    "Marital Status", ["married", "single", "divorced"]
)

education = st.sidebar.selectbox(
    "Education Level", ["secondary", "tertiary", "primary", "unknown"]
)

default = st.sidebar.selectbox("Credit in Default?", ["no", "yes"])

balance = st.sidebar.number_input("Average Yearly Balance (€)", value=1000)

housing = st.sidebar.selectbox("Housing Loan?", ["no", "yes"])

loan = st.sidebar.selectbox("Personal Loan?", ["no", "yes"])

contact = st.sidebar.selectbox(
    "Contact Communication Type", ["cellular", "telephone", "unknown"]
)

day = st.sidebar.slider("Last Contact Day of Month", 1, 31, 15)

month = st.sidebar.selectbox(
    "Last Contact Month",
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"]
)

duration = st.sidebar.number_input("Last Contact Duration (seconds)", value=180)

campaign = st.sidebar.number_input(
    "Number of Contacts during Campaign", min_value=1, value=1
)

pdays = st.sidebar.number_input(
    "Days since previous campaign contact (-1 = never)", value=-1
)

previous = st.sidebar.number_input(
    "Number of Contacts before Campaign", value=0
)

poutcome = st.sidebar.selectbox(
    "Outcome of Previous Marketing Campaign",
    ["unknown", "other", "failure", "success"]
)


# ============================================================
# 3. CREATE INPUT DATAFRAME
# ============================================================

input_dict = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}

input_df = pd.DataFrame([input_dict])

st.subheader("Customer Input Summary")
st.dataframe(input_df)


# ============================================================
# 4. ANN PREDICTION
# ============================================================

if st.button("Predict Subscription (ANN)", type="primary"):

    try:
        prediction = pipeline.predict(input_df)[0]

        proba = pipeline.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1 or str(prediction).lower() == "yes":

            st.success(
                f"🎉 High Likelihood to Subscribe! "
                f"(ANN Probability: {proba * 100:.2f}%)"
            )

        else:

            st.warning(
                f"⚠️ Unlikely to Subscribe. "
                f"(ANN Probability: {proba * 100:.2f}%)"
            )

    except Exception as err:
        st.error(f"Prediction failed: {err}")