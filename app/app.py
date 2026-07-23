import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer is likely to churn.")

# =============================
# Sidebar Inputs
# =============================

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male","Female"])

senior = st.sidebar.selectbox(
    "Senior Citizen",
    ["Yes","No"]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes","No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes","No"]
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes","No"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "No",
        "Yes",
        "No phone service"
    ]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

device = st.sidebar.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

tech = st.sidebar.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.number_input(
    "Monthly Charges",
    0.0,
    150.0,
    70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

# =============================
# Create Input DataFrame
# =============================

input_df = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phone],
    "MultipleLines":[multiple],
    "InternetService":[internet],
    "OnlineSecurity":[online_security],
    "OnlineBackup":[online_backup],
    "DeviceProtection":[device],
    "TechSupport":[tech],
    "StreamingTV":[tv],
    "StreamingMovies":[movies],
    "Contract":[contract],
    "PaperlessBilling":[paperless],
    "PaymentMethod":[payment],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total]
})

# One-Hot Encoding
input_df = pd.get_dummies(input_df)

# Match Training Columns
input_df = input_df.reindex(columns=features, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# =============================
# Prediction
# =============================

if st.button("Predict Churn"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric(
            "Probability",
            f"{probability*100:.2f}%"
        )

    with col2:
        st.metric(
            "Tenure",
            f"{tenure} Months"
        )

    with col3:
        st.metric(
            "Monthly Charges",
            f"${monthly:.2f}"
        )

    st.divider()

    if prediction==1:

        st.error("⚠ Customer is likely to Churn")

    else:

        st.success("✅ Customer is likely to Stay")

    st.progress(float(probability))

    st.subheader("Customer Summary")

    st.dataframe(input_df.T)