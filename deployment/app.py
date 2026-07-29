import streamlit as st
import pickle
import pandas as pd

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Revenue Prediction",
    page_icon="💰",
    layout="centered"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
try:
    with open("linear_regression_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("feature_columns.pkl", "rb") as file:
        feature_columns = pickle.load(file)

except FileNotFoundError:
    st.error("Model files not found.")
    st.stop()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("Project Information")

st.sidebar.write("**Project:** Customer Revenue Prediction")
st.sidebar.write("**Algorithm:** Linear Regression")
st.sidebar.write("**R² Score:** 0.9457")
st.sidebar.write("**Deployment:** Streamlit + Pickle")

st.sidebar.markdown("---")
st.sidebar.write("Developed by")
st.sidebar.write("**Agnibha Paul**")
st.sidebar.write("MCA Summer Training Project")

# ---------------------------------------------------
# Main Title
# ---------------------------------------------------
st.title("💰 Customer Revenue Prediction System")

st.write(
    "Predict customer revenue using a trained **Linear Regression** "
    "machine learning model."
)

st.markdown("---")

# ---------------------------------------------------
# Model Information
# ---------------------------------------------------
st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Algorithm", "Linear Regression")

with col2:
    st.metric("R² Score", "0.9457")

st.markdown("---")

# ---------------------------------------------------
# Customer Inputs
# ---------------------------------------------------
st.subheader("Customer Information")

total_quantity = st.number_input(
    "Total Quantity Purchased",
    min_value=0,
    value=100,
    step=1
)

average_price = st.number_input(
    "Average Unit Price (£)",
    min_value=0.0,
    value=5.00,
    step=0.01,
    format="%.2f"
)

number_of_transactions = st.number_input(
    "Number of Transactions",
    min_value=0,
    value=10,
    step=1
)

number_of_products = st.number_input(
    "Number of Unique Products",
    min_value=0,
    value=20,
    step=1
)

average_quantity = st.number_input(
    "Average Quantity Purchased",
    min_value=0.0,
    value=10.00,
    step=0.01,
    format="%.2f"
)

st.markdown("---")

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict Revenue", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            total_quantity,
            average_price,
            number_of_transactions,
            number_of_products,
            average_quantity
        ]],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed successfully!")

    st.metric(
        label="Predicted Customer Revenue",
        value=f"£{prediction:,.2f}"
    )

    st.info(
        "The predicted value represents the estimated revenue generated "
        "by a customer based on the provided purchasing behaviour."
    )