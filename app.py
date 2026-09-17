import streamlit as st
import pickle
import json
import pandas as pd
from datetime import datetime

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Revenue Prediction",
    page_icon="💰",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Card-style containers */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        text-align: center;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.4);
        margin-top: 16px;
    }

    .result-amount {
        font-size: 3.2rem;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -1px;
    }

    .result-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.75);
        margin-top: 6px;
    }

    /* Warning card */
    .warning-card {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.4);
        border-radius: 12px;
        padding: 16px;
        color: #fbbf24;
        margin-top: 12px;
    }

    /* Error card */
    .error-card {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 16px;
        color: #f87171;
        margin-top: 12px;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(167, 139, 250, 0.5), transparent);
        margin: 24px 0;
    }

    /* Streamlit metric override */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        color: #a78bfa !important;
    }

    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.6) !important;
    }

    /* Input labels */
    label {
        color: rgba(255, 255, 255, 0.85) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        border-right: 1px solid rgba(167, 139, 250, 0.2);
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 14px 28px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        color: rgba(255, 255, 255, 0.6) !important;
        border-radius: 8px !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(124, 58, 237, 0.4) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# Load Model & Metadata
# ---------------------------------------------------
@st.cache_resource
def load_model():
    with open("linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns


@st.cache_data
def load_metadata():
    with open("model_metadata.json", "r") as f:
        return json.load(f)


try:
    model, feature_columns = load_model()
    meta = load_metadata()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
with st.sidebar:
    st.markdown("## 💰 Revenue Predictor")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🤖 Model")
    st.markdown(f"**Algorithm:** {meta['algorithm']}")
    st.markdown(f"**R² Score:** `{meta['r2_score']}`")
    st.markdown(f"**MAE:** `£{meta['mae']:,.2f}`")
    st.markdown(f"**RMSE:** `£{meta['rmse']:,.2f}`")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📊 Dataset")
    st.markdown(f"**Source:** {meta['dataset']}")
    st.markdown(f"**Customers:** `{meta['n_customers']:,}`")
    st.markdown(f"**Features:** `{meta['n_features']}`")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 👤 About")
    st.markdown(f"**Developer:** {meta['author']}")
    st.markdown(f"**Project:** {meta['project']}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.caption("Built with Streamlit · scikit-learn")


# ---------------------------------------------------
# Main Title
# ---------------------------------------------------
st.markdown("# 💰 Customer Revenue Prediction")
st.markdown(
    "Predict estimated customer revenue using a trained **Linear Regression** model "
    "built on the Online Retail II dataset."
)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Predict Revenue", "📈 Model Performance"])

# ==================================================
# TAB 1 — Prediction
# ==================================================
with tab1:

    st.markdown("### Customer Behaviour Inputs")
    st.markdown(
        "Enter the customer's purchasing behaviour below. "
        "All fields are required."
    )
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_quantity = st.number_input(
            "🛒 Total Quantity Purchased",
            min_value=1,
            max_value=500_000,
            value=100,
            step=1,
            help="Total units purchased by this customer across all orders."
        )

    with col2:
        average_price = st.number_input(
            "💷 Average Unit Price (£)",
            min_value=0.01,
            max_value=10_000.0,
            value=5.00,
            step=0.01,
            format="%.2f",
            help="Average price per unit across all products purchased."
        )

    with col3:
        number_of_transactions = st.number_input(
            "🧾 Number of Transactions",
            min_value=1,
            max_value=10_000,
            value=10,
            step=1,
            help="Total number of distinct invoices/orders placed."
        )

    col4, col5 = st.columns(2)

    with col4:
        number_of_products = st.number_input(
            "📦 Unique Products Purchased",
            min_value=1,
            max_value=5_000,
            value=20,
            step=1,
            help="Number of distinct product types the customer has purchased."
        )

    with col5:
        average_quantity = st.number_input(
            "📊 Average Quantity per Order",
            min_value=0.01,
            max_value=50_000.0,
            value=10.00,
            step=0.01,
            format="%.2f",
            help="Average number of units purchased per transaction."
        )

    st.markdown("")

    # ---------------------------------------------------
    # Input Validation
    # ---------------------------------------------------
    validation_warnings = []

    if average_quantity > total_quantity:
        validation_warnings.append(
            "⚠️ Average quantity per order cannot exceed total quantity purchased."
        )

    if number_of_transactions > total_quantity:
        validation_warnings.append(
            "⚠️ Number of transactions cannot exceed total quantity purchased."
        )

    if number_of_products > number_of_transactions * 50:
        validation_warnings.append(
            "⚠️ Unique product count seems unusually high relative to transaction count."
        )

    for w in validation_warnings:
        st.markdown(f'<div class="warning-card">{w}</div>', unsafe_allow_html=True)

    if validation_warnings:
        st.markdown("")

    # ---------------------------------------------------
    # Predict Button
    # ---------------------------------------------------
    predict_clicked = st.button(
        "🔮 Predict Revenue",
        use_container_width=True,
        disabled=bool(validation_warnings)
    )

    if validation_warnings:
        st.caption("Fix the warnings above to enable prediction.")

    if predict_clicked:
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

        try:
            prediction = model.predict(input_data)[0]

            if prediction < 0:
                st.markdown(
                    '<div class="error-card">'
                    '⚠️ The model returned a negative revenue estimate (£{:,.2f}). '
                    'This input combination is outside the model\'s reliable range. '
                    'Try adjusting the values to better reflect realistic purchasing behaviour.'
                    '</div>'.format(prediction),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p class="result-label">Estimated Customer Revenue</p>
                        <p class="result-amount">£{prediction:,.2f}</p>
                        <p class="result-label" style="margin-top:12px;">
                            Based on {number_of_transactions} transaction(s) · 
                            {number_of_products} product(s) · 
                            {total_quantity:,} units purchased
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("")
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.metric("Predicted Revenue", f"£{prediction:,.2f}")
                with rc2:
                    st.metric("Revenue per Transaction", f"£{prediction / number_of_transactions:,.2f}")
                with rc3:
                    st.metric("Revenue per Unit", f"£{prediction / total_quantity:,.4f}")

                st.markdown("")
                st.info(
                    "💡 This prediction is based on the customer's purchasing behaviour features. "
                    "It estimates the total revenue this customer profile would generate. "
                    f"The model has an R² of {meta['r2_score']} and a mean absolute error of £{meta['mae']:,.2f}."
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ==================================================
# TAB 2 — Model Performance
# ==================================================
with tab2:

    st.markdown("### Model Evaluation Summary")
    st.markdown(
        "Four regression models were trained and compared. "
        "The best model by R² score was selected for deployment."
    )
    st.markdown("")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("R² Score", meta["r2_score"], help="Proportion of variance explained. Closer to 1.0 is better.")
    with m2:
        st.metric("MAE", f"£{meta['mae']:,.2f}", help="Mean Absolute Error — average prediction error in pounds.")
    with m3:
        st.metric("RMSE", f"£{meta['rmse']:,.2f}", help="Root Mean Squared Error — penalises large errors more.")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Model Comparison")

    comparison_data = {
        "Model": ["Linear Regression ✅", "Random Forest", "K-Nearest Neighbours", "Decision Tree"],
        "MAE (£)": [959.99, 656.37, 932.13, 999.67],
        "RMSE (£)": [4438.96, 7549.95, 8464.57, 10806.46],
        "R² Score": [0.9457, 0.8430, 0.8027, 0.6784],
        "Selected": ["✅ Yes", "—", "—", "—"]
    }

    df_compare = pd.DataFrame(comparison_data)
    st.dataframe(
        df_compare,
        use_container_width=True,
        hide_index=True,
        column_config={
            "MAE (£)": st.column_config.NumberColumn(format="£%.2f"),
            "RMSE (£)": st.column_config.NumberColumn(format="£%.2f"),
            "R² Score": st.column_config.ProgressColumn(
                min_value=0,
                max_value=1,
                format="%.4f"
            ),
        }
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Feature Description")
    feature_data = {
        "Feature": feature_columns,
        "Description": [
            "Sum of all units purchased by the customer",
            "Mean price per unit across all products",
            "Count of distinct invoices placed",
            "Count of distinct product types purchased",
            "Mean units per transaction",
        ]
    }
    st.dataframe(pd.DataFrame(feature_data), use_container_width=True, hide_index=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Known Limitations")
    st.markdown("""
    - **No demographic data** — country, customer age, or segment are not considered.
    - **No seasonality** — temporal patterns (holidays, peak seasons) are not captured.
    - **Outlier sensitivity** — a small number of very high-revenue customers may skew predictions.
    - **Retraining required** — model performance may degrade over time as purchasing behaviour evolves.
    - **No feature scaling** — KNN and other distance-based models may underperform due to raw feature magnitudes.
    """)
