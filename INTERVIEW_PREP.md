# 🎯 Customer Revenue Prediction — Complete Interview Preparation Guide

> **Written in plain English. No complicated jargon. Just clear explanations so you can explain every part of this project confidently.**

---

## 📋 TABLE OF CONTENTS

1. [Quick Brief (30-second pitch)](#1-quick-brief)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [The Dataset](#3-the-dataset)
4. [Step-by-Step: What the Code Does](#4-step-by-step-what-the-code-does)
   - Data Cleaning
   - Feature Engineering
   - Exploratory Data Analysis
   - Model Training
   - Model Evaluation
   - Saving the Model
5. [The Web App (app.py)](#5-the-web-app-apppy)
6. [All Important Functions and Logic Explained](#6-all-important-functions-and-logic-explained)
7. [Interview Questions — Low Level (Fresher/Intern)](#7-low-level-interview-questions)
8. [Interview Questions — Mid Level](#8-mid-level-interview-questions)
9. [Interview Questions — High Level (Senior/Deep Dive)](#9-high-level-interview-questions)
10. [Quick Revision Cheatsheet](#10-quick-revision-cheatsheet)

---

## 1. Quick Brief

> Say this when someone asks "tell me about your project" — keep it under 30 seconds.

"I built a machine learning web application that predicts how much revenue a customer is likely to generate, based on their purchasing behaviour. I used the Online Retail II dataset from UCI, which has over a million real UK e-commerce transactions. I cleaned the data, engineered customer-level features, trained and compared four regression models, and deployed the best one — Linear Regression with an R² of 0.9457 — as a Streamlit web app where you can enter customer data and get an instant revenue prediction."

---

## 2. The Problem We Are Solving

Imagine you run an online store. You have thousands of customers. Some buy a lot, some barely buy anything. You want to know **in advance** — how much money will a particular customer bring in?

If you know that, you can:
- Give loyalty rewards to high-value customers
- Send discount offers to low-value customers to bring them back
- Decide where to spend your marketing budget

**This is exactly what this project does.** Given a customer's buying habits (how much they buy, how often, what price range), the model predicts their total revenue.

---

## 3. The Dataset

**Name:** Online Retail II Dataset  
**Source:** UCI Machine Learning Repository  
**What it contains:** Real transaction records from a UK-based online gift shop between 2009 and 2011

### The raw columns:

| Column | What it means |
|---|---|
| `Invoice` | A unique ID for each order |
| `StockCode` | A code for each product |
| `Description` | Name of the product |
| `Quantity` | How many units were bought (negative = returned) |
| `InvoiceDate` | Date and time of the order |
| `Price` | Price per unit in British Pounds (£) |
| `Customer ID` | Unique ID for each customer |
| `Country` | Country of the customer |

**Raw size:** 1,067,371 rows × 8 columns  
**After cleaning:** 779,425 rows  
**After feature engineering:** 5,878 customers (one row per customer)

---

## 4. Step-by-Step: What the Code Does

### STEP 1 — Loading the Data

```python
import pandas as pd

df = pd.read_csv("online_retail_II.csv")
```

**What this does:** Reads the CSV file and puts all the data into a "table" called a DataFrame. Think of it like opening a spreadsheet in Python.

---

### STEP 2 — Exploring the Raw Data

```python
df.shape          # How many rows and columns?
df.info()         # What data types are in each column?
df.describe()     # Basic statistics like min, max, average
df.isnull().sum() # How many missing values?
```

**What we found:**
- 243,007 rows had no Customer ID (we can't track these customers — removed)
- 4,382 rows had no Description (minor issue)
- Quantity goes negative (returns/refunds — we removed these)
- Price goes negative (credits/adjustments — we removed these)
- 34,335 duplicate rows (exact same row appearing twice — removed)

---

### STEP 3 — Cleaning the Data

```python
# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where Customer ID is missing
df = df.dropna(subset=["Customer ID"])

# Convert Customer ID from decimal (12345.0) to integer (12345)
df["Customer ID"] = df["Customer ID"].astype(int)

# Convert date column from text to actual datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Keep only real purchases (no returns, no credits)
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
```

**Why we remove negatives:** Returns and credits are not revenue. Including them would give wrong numbers.

**Why convert Customer ID to int:** It was stored as `12345.0` (a decimal) which is odd for an ID. Converting to `12345` (integer) is cleaner and correct.

---

### STEP 4 — Creating the Revenue Column

```python
df["Revenue"] = df["Quantity"] * df["Price"]
```

**Simple math:** If a customer buys 5 items at £3 each, their revenue for that row is £15.

This is a **derived feature** — we calculated it from existing columns.

---

### STEP 5 — Feature Engineering (The Most Important Step)

Right now, every row is one product in one transaction. But we want **one row per customer** — a summary of everything they've ever bought.

```python
customer_df = df.groupby("Customer ID").agg(
    Total_Revenue        = ("Revenue",    "sum"),      # Total money spent
    Total_Quantity       = ("Quantity",   "sum"),      # Total units bought
    Average_Price        = ("Price",      "mean"),     # Average price they pay
    Number_of_Transactions = ("Invoice",  "nunique"),  # How many orders they placed
    Number_of_Products   = ("StockCode",  "nunique"),  # How many different products
    Average_Quantity     = ("Quantity",   "mean")      # Average units per order
).reset_index()
```

**What `groupby` does:** It groups all rows by Customer ID and then runs a calculation on each group.

**What `.agg()` does:** "agg" = aggregation. It says "for each customer, calculate these things":
- `"sum"` → add everything up
- `"mean"` → calculate the average
- `"nunique"` → count how many **unique** (different) values exist

**Result:** 5,878 rows — one per customer — with 6 columns describing their behaviour.

---

### STEP 6 — Exploratory Data Analysis (EDA)

EDA just means "look at the data visually before building a model." We made 6 charts:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Chart 1: Revenue distribution
sns.histplot(customer_df["Total_Revenue"], bins=50, kde=True)
```

**What we saw:**
- Most customers generate low revenue (under £1,000)
- A very small number of customers generate extremely high revenue (£50,000+)
- This is called a **right-skewed distribution** — the tail goes to the right
- More transactions = more revenue (obvious, but the data confirms it)
- The correlation heatmap showed `Total_Quantity` and `Number_of_Transactions` are most correlated with revenue

---

### STEP 7 — Preparing Data for Machine Learning

```python
from sklearn.model_selection import train_test_split

# X = the input features (what we feed into the model)
X = customer_df.drop(columns=["Customer ID", "Total_Revenue"])

# y = the target (what we want to predict)
y = customer_df["Total_Revenue"]

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Why split?** You train the model on 80% of the data, then check how well it performs on the other 20% that it has **never seen before**. This tells you if the model actually learned something useful or just memorised the training data.

**What is `random_state=42`?** It makes the split the same every time you run the code. If you didn't set this, you'd get a different split each time and your results would be inconsistent. The number 42 is just a convention (from "The Hitchhiker's Guide to the Galaxy" — it means nothing special).

**Training set size:** 4,702 customers  
**Testing set size:** 1,176 customers

---

### STEP 8 — Training the Models

We trained 4 models and compared them. Here's each one explained simply:

---

#### Model 1: Linear Regression

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
```

**What it does:** Draws the "best fit line" through the data. It finds a formula like:

```
Revenue = (a × Total_Quantity) + (b × Average_Price) + (c × Transactions) + ... + constant
```

It figures out the best values for `a`, `b`, `c` etc. by minimising the error between predicted and actual values.

**Think of it like:** Drawing a straight line through a scatter plot that gets as close to all points as possible.

---

#### Model 2: Decision Tree

```python
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
```

**What it does:** Creates a flowchart of yes/no questions.

```
Is Total_Quantity > 500?
  YES → Is Number_of_Transactions > 20? → ...
  NO  → Is Average_Price > 10? → ...
```

**Problem:** Without a `max_depth` limit, it keeps splitting until it memorises the training data perfectly — but then performs poorly on new data. This is called **overfitting**.

---

#### Model 3: Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

**What it does:** Builds 100 decision trees (`n_estimators=100`), each trained on a slightly different random sample of the data. The final prediction is the **average** of all 100 trees.

**Why it's better than one tree:** One tree can be wrong. 100 trees that each make small errors, when averaged, tend to cancel each other out and give a better result. This is called **ensemble learning**.

---

#### Model 4: K-Nearest Neighbours (KNN)

```python
from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
```

**What it does:** When predicting revenue for a new customer, it finds the 5 most similar customers in the training data and averages their revenue.

**Think of it like:** "Who are the 5 customers most similar to this one? What was their revenue? Let's average that."

**Problem here:** The features have very different scales. `Total_Quantity` can be 50,000 while `Average_Price` is £3. KNN uses distance calculations — so the big numbers completely dominate. We never scaled the features, which hurt KNN's performance.

---

### STEP 9 — Evaluating the Models

After training, we test each model on the test set (the 20% it never saw):

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = lr.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
```

**The three metrics explained simply:**

| Metric | What it means | Our LR Score |
|---|---|---|
| **MAE** (Mean Absolute Error) | On average, how many £ off is the prediction? | £959.99 |
| **RMSE** (Root Mean Squared Error) | Like MAE but punishes big mistakes more | £4,438.96 |
| **R² Score** | What % of the variation in revenue does the model explain? (1.0 = perfect) | 0.9457 |

**Example of MAE:** If you predict £5,000 revenue for a customer, and the real answer is £4,500, the error is £500. The MAE is the average of all such errors across all test customers.

**R² = 0.9457 means:** The model explains 94.57% of why customers have different revenue levels. That's very good.

---

### Final Model Comparison Table

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression ✅** | £959.99 | £4,438.96 | **0.9457** |
| Random Forest | £656.37 | £7,549.95 | 0.8430 |
| K-Nearest Neighbours | £932.13 | £8,464.57 | 0.8027 |
| Decision Tree | £999.67 | £10,806.46 | 0.6784 |

**Why we chose Linear Regression:** It has the highest R² (0.9457). Even though Random Forest has a lower MAE, R² is a better overall measure of how well the model fits the data.

---

### STEP 10 — Saving the Model

```python
import pickle

# Save the trained model
with open("linear_regression_model.pkl", "wb") as file:
    pickle.dump(lr, file)

# Save the column names (so we know what order to feed inputs)
feature_columns = list(X.columns)
with open("feature_columns.pkl", "wb") as file:
    pickle.dump(feature_columns, file)
```

**What is pickle?** Pickle is Python's way of converting any object (like a trained model) into a file that you can save and load later. Like saving a game — you can pick it up where you left off without retraining.

**Why save `feature_columns`?** The model was trained with features in a specific order: `[Total_Quantity, Average_Price, Number_of_Transactions, Number_of_Products, Average_Quantity]`. When we load the model and give it new data, the order must be exactly the same or the model will make wrong predictions.

**`.pkl` = Pickle file format.** `"wb"` = write binary (the only mode that works for pickle). `"rb"` = read binary (used when loading).

---

## 5. The Web App (app.py)

The app is built with **Streamlit** — a Python library that turns Python scripts into web applications with no HTML or JavaScript needed.

### How it works:

```python
import streamlit as st
import pickle
import json
import pandas as pd
```

**Load the model once and cache it:**

```python
@st.cache_resource
def load_model():
    with open("linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns
```

**`@st.cache_resource`** — This decorator tells Streamlit: "Load the model once when the app starts. Don't reload it every time the user clicks something." This makes the app much faster.

---

**Load model metadata from JSON:**

```python
@st.cache_data
def load_metadata():
    with open("model_metadata.json", "r") as f:
        return json.load(f)
```

`model_metadata.json` stores the R², MAE, RMSE, and other info. The app reads this instead of hardcoding values — so if the model is retrained, the app automatically shows the correct updated metrics.

---

**Input fields:**

```python
total_quantity = st.number_input(
    "🛒 Total Quantity Purchased",
    min_value=1,
    max_value=500_000,
    value=100,
    step=1,
    help="Total units purchased by this customer across all orders."
)
```

`st.number_input` creates a number box. `min_value` and `max_value` prevent nonsensical inputs. `value` is the default. `help` shows a tooltip.

---

**Input Validation:**

```python
validation_warnings = []

if average_quantity > total_quantity:
    validation_warnings.append("⚠️ Average quantity per order cannot exceed total quantity purchased.")

if number_of_transactions > total_quantity:
    validation_warnings.append("⚠️ Number of transactions cannot exceed total quantity purchased.")
```

Before allowing a prediction, we check if the inputs make logical sense. If average quantity is higher than total quantity — that's impossible. We show a warning and **disable the predict button** until it's fixed.

---

**Making the Prediction:**

```python
input_data = pd.DataFrame(
    [[total_quantity, average_price, number_of_transactions, number_of_products, average_quantity]],
    columns=feature_columns
)

prediction = model.predict(input_data)[0]
```

We put the user's inputs into a DataFrame with the **same column names** the model was trained on. Then `.predict()` returns an array — we take index `[0]` to get the single number.

---

**Handling negative predictions:**

```python
if prediction < 0:
    st.error("The model returned a negative revenue estimate...")
else:
    st.success(f"Predicted Revenue: £{prediction:,.2f}")
```

Linear Regression can mathematically return negative numbers for weird inputs. We catch this and show a warning instead of displaying "£-500.00" which would confuse users.

---

## 6. All Important Functions and Logic Explained

### `df.drop_duplicates()`
Removes rows that are exactly identical. Simple.

### `df.dropna(subset=["Customer ID"])`
Removes rows where `Customer ID` is missing (NaN). We can't track anonymous customers.

### `df.astype(int)`
Changes data type. `12345.0 → 12345`. Useful for IDs that got read as floats.

### `pd.to_datetime()`
Converts a string like `"2010-12-01 08:26:00"` into a proper date object Python can work with.

### `df.groupby().agg()`
Groups rows by a column and applies calculations. The core of feature engineering.

```python
# Each Customer ID becomes one row
# sum → add all values
# mean → find the average
# nunique → count unique values
df.groupby("Customer ID").agg(
    Total_Revenue = ("Revenue", "sum"),
    Average_Price = ("Price", "mean"),
    Number_of_Transactions = ("Invoice", "nunique")
)
```

### `train_test_split(X, y, test_size=0.2, random_state=42)`
Splits data into training (80%) and testing (20%) sets. `random_state` fixes the randomness for reproducibility.

### `model.fit(X_train, y_train)`
This is how a model "learns." It looks at the training data and adjusts its internal parameters to minimise prediction error.

### `model.predict(X_test)`
Uses the learned parameters to make predictions on new data. Returns an array of predicted values.

### `mean_absolute_error(y_test, y_pred)`
Calculates average error: `mean(|actual - predicted|)`

### `mean_squared_error(y_test, y_pred)`
Squares the errors before averaging. Big errors count much more. `sqrt()` to get RMSE.

### `r2_score(y_test, y_pred)`
Measures how much of the variance in the target the model explains. Range: 0 to 1. Higher is better.

### `pickle.dump(model, file)` / `pickle.load(file)`
Save and load any Python object to/from a file.

### `@st.cache_resource`
Streamlit decorator. Runs the function once, stores the result in memory. Used for heavy objects like ML models.

### `@st.cache_data`
Same idea but for regular data (DataFrames, dicts, etc.)

### `pd.DataFrame([[v1, v2, v3]], columns=[...])`
Creates a one-row DataFrame from a list of values. Used to format user inputs for model prediction.

---

## 7. Low Level Interview Questions

> These are for freshers, interns, or anyone just starting out.

---

**Q1: What is this project about?**

A: It predicts how much revenue a customer will generate based on their buying behaviour. We used a Linear Regression model trained on real e-commerce data. The result is deployed as a web app.

---

**Q2: What is machine learning in simple words?**

A: It's teaching a computer to find patterns in data and make predictions, without explicitly programming every rule. Instead of saying "if a customer buys more than 100 items, their revenue is high", you show the computer thousands of examples and let it figure out the pattern itself.

---

**Q3: What is Linear Regression?**

A: It finds the best straight line through data. It learns a formula like:  
`Revenue = a × Quantity + b × Price + c × Transactions + ...`  
The model figures out the best values for `a`, `b`, `c` to minimise prediction error.

---

**Q4: What is a DataFrame?**

A: A DataFrame is a table in Python (from the pandas library). Like a spreadsheet — it has rows, columns, and labels. It's the standard way to store and manipulate data in Python.

---

**Q5: What does `train_test_split` do?**

A: It splits your data into two parts. You train the model on the larger part (80%) and test it on the smaller part (20%). The test data is data the model never saw during training — so it honestly measures how well the model predicts new, unseen data.

---

**Q6: What is `random_state=42`?**

A: It fixes the random seed so the split is the same every time you run the code. Without it, you'd get a different split each run and your results would change. 42 is just a commonly used arbitrary number.

---

**Q7: What is a `.pkl` file?**

A: A pickle file. Pickle is Python's built-in way to save any object (like a trained model) to disk. Later, you can load it back into memory without retraining. It's like saving your progress in a game.

---

**Q8: What is R² score?**

A: R² (R-squared) tells you how well the model explains the variation in the data. A score of 1.0 means perfect predictions. A score of 0 means the model is no better than just guessing the average every time. Our model scored 0.9457 — meaning it explains 94.57% of the variation in customer revenue.

---

**Q9: What is the difference between MAE and RMSE?**

A: Both measure prediction error.  
- **MAE** = average of all errors (all errors count equally)  
- **RMSE** = average of squared errors, then square-rooted (big errors count more)  

RMSE penalises large mistakes more heavily. If you care more about avoiding huge errors, use RMSE.

---

**Q10: What is Streamlit?**

A: A Python library that turns Python scripts into web apps automatically. No HTML, CSS, or JavaScript needed. You write `st.number_input(...)` and Streamlit creates a proper web form element for you.

---

**Q11: What is feature engineering?**

A: Transforming raw data into useful input variables for the model. In this project, we had millions of transaction rows but we wanted to predict revenue per customer. So we grouped all transactions by customer and calculated summaries (total quantity, avg price, etc.). These summaries are the "engineered features."

---

**Q12: Why did you remove rows where Quantity < 0?**

A: Negative quantities represent returns and refunds — a customer sent something back. These are not purchases, so they don't contribute to revenue. Including them would corrupt the model.

---

**Q13: What is `groupby` in pandas?**

A: It groups rows that share the same value in a column and lets you apply functions to each group. Like: "for each Customer ID, add up all their Revenue." The result is one row per customer.

---

**Q14: What is the difference between supervised and unsupervised learning?**

A: **Supervised:** You have both inputs (X) and the correct answer (y). The model learns the relationship. Our project is supervised — we know each customer's actual total revenue.  
**Unsupervised:** You only have inputs. The model finds patterns on its own (like clustering similar customers together).

---

**Q15: What is overfitting?**

A: When a model performs great on training data but poorly on new data. It "memorised" the training data instead of learning general patterns. The Decision Tree in this project overfitted badly — it got perfect training scores but R² = 0.6784 on test data.

---

## 8. Mid Level Interview Questions

> These go deeper. You need to understand *why*, not just *what*.

---

**Q1: Why did Linear Regression outperform Random Forest despite Random Forest having a lower MAE?**

A: We selected the model based on R² score, where Linear Regression scored 0.9457 vs Random Forest's 0.8430. R² measures overall explanatory power — how much of the variance in the target the model captures. The revenue data appears to have a strong underlying linear relationship between the features and Total_Revenue, which Linear Regression captures well. Random Forest's lower MAE (£656 vs £960) suggests it handles individual outliers better, but its significantly lower R² (0.84 vs 0.94) means it captures less of the overall variance. For a business use case where understanding the overall revenue pattern matters more than minimising individual prediction error, R² is the better selection criterion.

---

**Q2: What is multicollinearity and is it present in this project?**

A: Multicollinearity is when two or more input features are highly correlated with each other. In this project, `Total_Quantity` and `Average_Quantity` are both derived from the same `Quantity` column — so they're likely correlated. Similarly, `Number_of_Transactions` and `Total_Quantity` are probably correlated (more orders → more total quantity). Multicollinearity doesn't harm prediction accuracy in Linear Regression but makes the coefficients unreliable for interpretation — you can't say "increasing quantity by 1 unit increases revenue by £X" because the features are not independent.

---

**Q3: What is feature scaling and why didn't you apply it here?**

A: Feature scaling normalises all features to the same range (e.g., 0 to 1 or mean=0, std=1). It's important for distance-based algorithms like KNN and gradient-based algorithms like neural networks. Linear Regression doesn't mathematically require scaling because it fits a linear equation — the coefficients adjust for different scales automatically. That said, not scaling KNN likely hurt its performance. `Total_Quantity` can reach 50,000 while `Average_Price` is around £3 — KNN's distance calculations would be dominated by `Total_Quantity`, essentially ignoring the other features.

---

**Q4: What is the difference between `@st.cache_resource` and `@st.cache_data`?**

A: Both tell Streamlit to cache (store) the function's result so it doesn't run again on every page reload:
- `@st.cache_resource` — for objects that should be shared across all users and sessions, like ML models, database connections. The object is loaded once and reused.
- `@st.cache_data` — for data that can be copied safely, like DataFrames or dictionaries. Each user gets their own copy.

We use `@st.cache_resource` for the model (one model shared by all users) and `@st.cache_data` for the metadata JSON.

---

**Q5: Why did you save `feature_columns.pkl` separately?**

A: The trained model expects inputs in a specific column order. When we call `model.predict(input_data)`, the DataFrame must have columns in the exact same order as when the model was trained: `[Total_Quantity, Average_Price, Number_of_Transactions, Number_of_Products, Average_Quantity]`. By saving and loading `feature_columns`, we guarantee this order is always correct, even if someone refactors the code. It prevents a subtle but dangerous bug where columns get shuffled and the model silently makes wrong predictions.

---

**Q6: What are the limitations of a single train/test split?**

A: A single 80/20 split means your model's evaluation depends heavily on which 20% happened to end up in the test set. If the test set contains unusually easy or hard customers by chance, your metrics are misleading. Cross-validation solves this — it splits the data into K folds, trains on K-1 folds, tests on the remaining fold, and repeats K times. The final score is the average across all folds, which is a much more reliable estimate of true model performance.

---

**Q7: The revenue distribution is heavily right-skewed. How does that affect the model?**

A: Linear Regression minimises Mean Squared Error (MSE), which squares the errors. A customer with £100,000 revenue who is predicted at £80,000 contributes a squared error of 400,000,000 — vastly more than a typical customer. This means the model is disproportionately influenced by the handful of very high-revenue customers. The model may perform well on high-revenue customers and poorly on typical customers. A standard solution is to log-transform the target: `y = log(Total_Revenue)`, train the model, then back-transform predictions with `exp()`. This balances the influence of all customers.

---

**Q8: What does `nunique()` do in the aggregation and why is it useful?**

A: `nunique()` counts the number of **unique (distinct) values** in a column. When we write `("Invoice", "nunique")`, it counts how many different invoice numbers a customer has — which equals their number of distinct orders. If we used `count()` instead, we'd count every row (every product line), not every distinct order. This distinction matters: a customer who ordered once with 10 products would have `count=10` but `nunique=1`.

---

**Q9: What is the difference between `.fit()`, `.predict()`, and `.score()`?**

A: 
- `.fit(X_train, y_train)` — **Training phase.** The model looks at the training data and adjusts its internal parameters (coefficients in Linear Regression) to minimise prediction error.
- `.predict(X_test)` — **Inference phase.** Uses the learned parameters to make predictions on new data. Returns an array of predicted values.
- `.score(X_test, y_test)` — **Quick evaluation.** For regression models, returns the R² score directly. Equivalent to `r2_score(y_test, model.predict(X_test))`.

---

**Q10: Why is the Decision Tree's performance so bad?**

A: The Decision Tree was created with no `max_depth` parameter, meaning it kept splitting until it perfectly classified every training sample. This is called a **fully grown tree** and it massively overfits. On training data it likely got near-perfect scores, but on test data the R² was only 0.6784. A pruned tree (with `max_depth=5` or `min_samples_leaf=20`) would generalise much better.

---

**Q11: What is ensemble learning? How does Random Forest use it?**

A: Ensemble learning combines multiple models to get better performance than any single model. Random Forest is a **bagging** ensemble — it:
1. Creates 100 decision trees (`n_estimators=100`)
2. Each tree is trained on a random sample of the training data (with replacement — called "bootstrapping")
3. Each tree also uses a random subset of features at each split
4. Final prediction = average of all 100 trees' predictions

Errors that are random cancel out when averaged. Systematic biases from individual trees are reduced. The result is a more robust, generalisable model.

---

**Q12: What is the difference between regression and classification?**

A: 
- **Regression** — predicting a continuous number (£5,432.16 revenue)
- **Classification** — predicting a category ("High", "Medium", "Low" revenue, or "will buy again: Yes/No")

This project is **regression** because revenue is a continuous number, not a category.

---

## 9. High Level Interview Questions

> These are for senior roles, technical deep dives, or when someone wants to stress-test your knowledge.

---

**Q1: Why does Linear Regression work so well here (R²=0.9457)?**

A: The feature engineering step is key. We aggregated transaction data into customer-level summaries. At this aggregated level, the relationship between features and total revenue is genuinely linear: a customer who buys twice as many units at the same average price will generate roughly twice the revenue. This linearity is inherent in the formula `Revenue = Quantity × Price`. The groupby aggregation "smooths out" the noise from individual transactions, making the signal clean and linear. The model is essentially rediscovering a near-arithmetic relationship, which is why it performs so well.

---

**Q2: If you had to improve this model, what would you do first?**

A: In order of impact:

1. **Log-transform the target** — `y = log1p(Total_Revenue)`. Revenue is right-skewed. Log transformation makes the distribution normal, removes the disproportionate influence of whale customers, and typically improves linear model performance significantly.

2. **Cross-validation** — Replace the single split with `cross_val_score(lr, X, y, cv=5)` to get reliable performance estimates.

3. **Hyperparameter tuning for Random Forest** — Use `GridSearchCV` with `n_estimators`, `max_depth`, `min_samples_leaf`. Random Forest had the best MAE; with tuning it might win on R² too.

4. **Feature scaling for KNN** — Apply `StandardScaler` before KNN. Its poor performance (R²=0.80) is partly an artefact of unscaled features.

5. **Outlier treatment** — Winsorise or cap the top 1% of revenue customers to reduce their leverage on the model.

6. **VIF analysis** — Check for multicollinearity between `Total_Quantity`, `Average_Quantity`, and `Number_of_Transactions`. Remove or combine redundant features.

---

**Q3: What are the coefficients of Linear Regression and how would you interpret them?**

A: After training, `lr.coef_` gives you an array of coefficients for each feature. For example:

```
Total_Quantity:          +0.85  → Each additional unit purchased adds £0.85 to revenue
Average_Price:           +312.4 → Each £1 increase in avg price adds £312 to revenue
Number_of_Transactions:  +45.2  → Each additional order adds £45 to revenue
Number_of_Products:      +8.9   → Each additional unique product adds £9 to revenue
Average_Quantity:        -1.2   → Counter-intuitive — needs investigation
```

However, because of multicollinearity, these coefficients are not reliably interpretable in isolation. The coefficient for one feature absorbs effects from correlated features.

---

**Q4: How would you handle the right-skewed revenue distribution properly?**

A: Apply a log transformation to the target before training:

```python
import numpy as np

y_log = np.log1p(y)  # log1p = log(1 + y), handles zero values safely

X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

lr.fit(X_train, y_log_train)
y_log_pred = lr.predict(X_test)

# Back-transform predictions to original scale
y_pred = np.expm1(y_log_pred)  # expm1 = exp(x) - 1, inverse of log1p

# Evaluate on original scale
r2 = r2_score(y_test, y_pred)
```

This would likely improve R² by treating all customers equally regardless of their revenue magnitude.

---

**Q5: What is the bias-variance tradeoff and how does it manifest in this project?**

A: Every model has two sources of error:
- **Bias** — error from wrong assumptions. A model that's too simple (underfitting). Low bias = model can capture complex patterns.
- **Variance** — error from sensitivity to small changes in training data. A model that's too complex (overfitting). Low variance = model is stable and generalises.

In this project:
- **Decision Tree (fully grown)** — Very low bias (fits training data perfectly) but very high variance (completely different on test data). R² went from ~1.0 on training to 0.6784 on test. Classic overfitting.
- **Linear Regression** — Moderate bias (assumes linearity) but very low variance (stable predictions). Wins because the data genuinely has a linear structure.
- **Random Forest** — Reduces variance by averaging 100 trees, but its R² (0.8430) is still lower than Linear Regression — possibly because the ensemble introduces its own overhead in a dataset that's fundamentally linear.

---

**Q6: How would you deploy this to production properly?**

A: The current Streamlit app is fine for demos but not production-grade. For production:

1. **API layer** — Wrap the model in a FastAPI or Flask REST API. The `/predict` endpoint accepts JSON, validates inputs, runs prediction, returns JSON response.
2. **Containerisation** — Package everything in Docker. `Dockerfile` installs dependencies and starts the API. Ensures consistent behaviour across environments.
3. **Model versioning** — Use MLflow or DVC to track model versions, training data versions, and metrics. Never overwrite a model file directly.
4. **Input schema validation** — Use Pydantic models to enforce input types and ranges at the API layer.
5. **Monitoring** — Track prediction distributions over time. If the average predicted revenue drifts significantly, it signals the model needs retraining (data drift).
6. **CI/CD pipeline** — Automated tests run on every commit. If model performance on a held-out validation set drops below threshold, deployment is blocked.

---

**Q7: What is data leakage and could it be present here?**

A: Data leakage is when information from the test set (or future data) accidentally influences model training, making the model look better than it actually is.

In this project, there's a subtle potential leakage risk: `Total_Quantity` and `Average_Quantity` are both computed from the same `Quantity` column. More importantly, `Total_Revenue` (the target) = `Total_Quantity × Average_Price`. So the features contain arithmetic components of the target. The model is partially learning the definitional formula rather than discovering a genuinely predictive signal.

This doesn't make the model useless — knowing a customer's quantity and price behaviour IS useful for predicting their revenue. But it means the R²=0.9457 is partly inflated by this near-tautological relationship.

---

**Q8: How would you detect if the model's performance degrades over time in production?**

A: This is called **model drift monitoring**:

1. **Prediction drift** — Track the distribution of predictions over time. If the mean or distribution shifts significantly, alert.
2. **Feature drift** — Monitor the distribution of input features. If `Average_Price` suddenly averages £50 (vs £5 during training), the model is seeing out-of-distribution data.
3. **Actual vs predicted** — When ground truth becomes available (actual revenue after a period), compute MAE and R² on recent data and compare to baseline.
4. **Scheduled retraining** — Retrain on a rolling window of data every N months regardless of detected drift.

Tools: **Evidently AI**, **WhyLabs**, **MLflow**, or simple Python scripts logging to a database.

---

**Q9: Why did you use `pickle` instead of `joblib` for model serialisation?**

A: `joblib` is actually the recommended way to save scikit-learn models because it's more efficient for large numpy arrays (which models internally use). We used `pickle` for simplicity — and `joblib` is even listed in `requirements.txt` but not used (a known gap in the project). For a proper setup it should be:

```python
import joblib

joblib.dump(lr, "linear_regression_model.pkl")
model = joblib.load("linear_regression_model.pkl")
```

`joblib` uses memory-mapped files and compresses better, making it faster for models with large internal arrays like Random Forest.

---

**Q10: What would you change about the feature engineering step?**

A: Several improvements:

1. **Recency** — How recently did the customer last purchase? RFM analysis (Recency, Frequency, Monetary) is the standard approach for customer value prediction. Recency is missing here.

2. **Customer tenure** — How long have they been a customer? `max(date) - min(date)` per customer.

3. **Price variance** — Does the customer only buy cheap items or a mix? Standard deviation of price per customer.

4. **Return rate** — What fraction of their purchases did they return? Negative quantities divided by total. High return rates are a risk signal.

5. **Country** — One-hot encode the country. UK customers vs international customers may have very different revenue profiles.

These additions would likely improve R² further and make the model more robust.

---

## 10. Quick Revision Cheatsheet

> Read this the morning of your interview.

```
PROJECT TYPE:   Supervised ML — Regression
GOAL:           Predict Total_Revenue per customer
DATASET:        Online Retail II (UCI) — 1M+ UK e-commerce transactions
CLEANED TO:     779,425 rows
ENGINEERED TO:  5,878 customers × 5 features

5 FEATURES:
  1. Total_Quantity         — sum of all units bought
  2. Average_Price          — mean price per unit
  3. Number_of_Transactions — count of distinct invoices
  4. Number_of_Products     — count of distinct products
  5. Average_Quantity       — mean units per order

TARGET: Total_Revenue (continuous number — £)

MODELS TRAINED: 4 (Linear Regression, Decision Tree, Random Forest, KNN)
WINNER:         Linear Regression — R²=0.9457, MAE=£959.99, RMSE=£4,438.96

WHY LR WON:     Revenue = Quantity × Price is inherently linear.
                Aggregated customer-level data is clean and linear.

DEPLOYMENT:     Streamlit web app (app.py)
MODEL SAVED:    pickle (.pkl)
METADATA:       model_metadata.json (auto-loaded by app — no hardcoding)

KEY DECISIONS:
  - Removed returns (Quantity < 0) and credits (Price < 0)
  - groupby + agg to collapse 779k rows → 5,878 customer rows
  - 80/20 train-test split with random_state=42
  - feature_columns.pkl saved to preserve column order
  - Input validation in app prevents impossible inputs
  - Negative prediction guard shows warning instead of £-X

KNOWN GAPS (be honest about these):
  - No feature scaling (hurts KNN)
  - No cross-validation (single split)
  - No log transform on skewed target
  - No outlier treatment
  - No hyperparameter tuning
  - Decision Tree not pruned (overfits badly)
  - pickle used instead of joblib

KEY METRICS:
  R²   = proportion of variance explained (1.0 = perfect)
  MAE  = average absolute error in £
  RMSE = like MAE but squares errors first (penalises big mistakes)
```

---

*Prepared by Agnibha Paul — MCA Summer Training Project*  
*Customer Revenue Prediction using Online Retail II Dataset*
