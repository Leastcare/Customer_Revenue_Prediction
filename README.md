# 💰 Customer Revenue Prediction

A machine learning web application that predicts estimated customer revenue based on purchasing behaviour, built with scikit-learn and deployed using Streamlit.

---

## 📌 Project Overview

This project was developed as part of an **MCA Summer Training** programme. It trains and compares four regression models on the [Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI), selects the best-performing model, and deploys it as an interactive web application.

**Developer:** Agnibha Paul

---

## 🧠 Machine Learning Pipeline

### Dataset
- **Source:** Online Retail II (UCI Machine Learning Repository)
- **Raw size:** ~1,067,371 transactions × 8 columns
- **After cleaning:** 779,425 rows (returns and credits excluded)
- **After feature engineering:** 5,878 customers × 5 features + 1 target

### Data Cleaning Steps
1. Remove duplicate rows (34,335 removed)
2. Drop rows with missing `Customer ID` (~235,151 rows)
3. Cast `Customer ID` to integer
4. Parse `InvoiceDate` as datetime
5. Filter out returns and credits (`Quantity > 0` and `Price > 0`)
6. Create `Revenue = Quantity × Price`

### Feature Engineering
Transaction-level data is aggregated to one row per customer:

| Feature | Description |
|---|---|
| `Total_Quantity` | Sum of all units purchased |
| `Average_Price` | Mean price per unit |
| `Number_of_Transactions` | Count of distinct invoices |
| `Number_of_Products` | Count of distinct products purchased |
| `Average_Quantity` | Mean units per transaction |

### Model Comparison

| Model | MAE (£) | RMSE (£) | R² Score |
|---|---|---|---|
| **Linear Regression** ✅ | 959.99 | 4,438.96 | **0.9457** |
| Random Forest | 656.37 | 7,549.95 | 0.8430 |
| K-Nearest Neighbours | 932.13 | 8,464.57 | 0.8027 |
| Decision Tree | 999.67 | 10,806.46 | 0.6784 |

**Selected model:** Linear Regression (highest R² score of 0.9457)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Customer_Revenue_Prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
Customer_Revenue_Prediction/
│
├── app.py                          # Streamlit web application
├── Customer_Revenue_Prediction.ipynb  # Full ML pipeline notebook
├── linear_regression_model.pkl     # Serialised trained model
├── feature_columns.pkl             # Feature column order
├── model_metadata.json             # Model metrics and metadata
├── online_retail_II.csv            # Raw dataset (not committed)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🖥️ Application Features

- **Prediction tab:** Enter 5 customer behaviour inputs and get an instant revenue estimate
- **Validation:** Input consistency checks with clear warnings before prediction
- **Negative prediction guard:** Alerts the user if the model returns an unrealistic value
- **Model performance tab:** Full model comparison table, feature descriptions, and known limitations
- **Live metadata:** Model metrics loaded from `model_metadata.json` — always in sync with the deployed model

---

## 📊 App Preview

The app is split into two tabs:

**🔮 Predict Revenue**
- Input fields for all 5 customer behaviour features
- Real-time input validation
- Prediction result card with derived metrics (revenue per transaction, revenue per unit)

**📈 Model Performance**
- Key metrics (R², MAE, RMSE)
- Full model comparison table with progress bars
- Feature descriptions
- Known model limitations

---

## ⚠️ Known Limitations

- No demographic or geographic features (country, customer segment)
- No seasonality or time-series components
- Model may need retraining as purchasing behaviour evolves over time
- Outlier customers (very high revenue) may influence predictions disproportionately

---

## 📦 Dependencies

| Package | Version |
|---|---|
| streamlit | 1.60.0 |
| pandas | 3.0.5 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| seaborn | >=0.12.0 |
| joblib | 1.5.3 |

---

## 📄 License

This project is for educational purposes as part of an MCA Summer Training programme.
