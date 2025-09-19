import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import miceforest as mf
import plotly.express as px

# Constants
DATA_URL = "./data/credit_card_attrition_dataset_justin.csv"

# Config
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load the model
class ImputeMissing(BaseEstimator, TransformerMixin):

    def __init__(self, iterations=3, random_state=None):
        self.iterations = iterations
        self.random_state = random_state
        self.kernel = None

    def fit(self, X, y=None):
        X = X.reset_index(drop=True, inplace=False)
        self.kernel = mf.ImputationKernel(X, random_state=self.random_state)
        self.kernel.mice(self.iterations)

        return self

    def transform(self, X):
        if self.kernel is None:
            raise RuntimeError("Fit the imputer first.")
        X = X.reset_index(drop=True, inplace=False)
        print(X)

        return self.kernel.impute_new_data(X).complete_data()


with open("final_model.pickle", "rb") as file:
    final_model = pickle.load(file)
    file.close()


def predict(X):
    X["Age"] = X["Age"].astype("float64")
    X["Gender"] = X["Gender"].astype("category")
    X["Income"] = X["Income"].astype("float64")
    X["CreditLimit"] = X["CreditLimit"].astype("float64")
    X["TotalTransactions"] = X["TotalTransactions"].astype("float64")
    X["TotalSpend"] = X["TotalSpend"].astype("float64")
    X["Tenure"] = X["Tenure"].astype("float64")
    X["MaritalStatus"] = X["MaritalStatus"].astype("category")
    X["EducationLevel"] = X["EducationLevel"].astype("category")
    X["CardType"] = X["CardType"].astype("category")
    X["Country"] = X["Country"].astype("category")
    for col, categories in CATEGORICAL_CATEGORIES.items():
        X[col] = pd.Categorical(X[col], categories=categories)

    pred = final_model.predict(X).item()
    if pred == 1:
        return "The customer will churn."
    else:
        return "The customer will NOT churn."


def get_prediction():
    to_predict = pd.DataFrame({
        "Age": [st.session_state.predict_age],
        "Gender": [st.session_state.predict_gender],
        "Income": [st.session_state.predict_income],
        "CreditLimit": [st.session_state.predict_credit_limit],
        "TotalTransactions": [st.session_state.predict_transactions],
        "TotalSpend": [st.session_state.predict_spend],
        "Tenure": [st.session_state.predict_tenure],
        "MaritalStatus": [st.session_state.predict_marital_status],
        "EducationLevel": [st.session_state.predict_education],
        "CardType": [st.session_state.predict_card_type],
        "Country": [st.session_state.predict_country]
    })
    result = predict(to_predict)

    if result == 1:
        st.session_state.prediction = result
    else:
        st.session_state.prediction = result


# Loading dataset
def load_data(drop_na=True):
    dataset = pd.read_csv(DATA_URL)

    if drop_na:
        dataset.dropna(inplace=True)

    dataset["AttritionFlagNum"] = dataset["AttritionFlag"]
    dataset["AttritionFlag"] = dataset["AttritionFlag"].map({0: "Active", 1: "Churned"})

    return dataset


data = load_data()

# Default Filter Values
FILTER_DEFAULT_MIN_AGE = data["Age"].min()
FILTER_DEFAULT_MAX_AGE = data["Age"].max()
FILTER_DEFAULT_MIN_INCOME = data["Income"].min()
FILTER_DEFAULT_MAX_INCOME = data["Income"].max()
FILTER_DEFAULT_MIN_CREDIT_LIMIT = data["CreditLimit"].min()
FILTER_DEFAULT_MAX_CREDIT_LIMIT = data["CreditLimit"].max()
FILTER_DEFAULT_MIN_TRANSACTIONS = data["TotalTransactions"].min()
FILTER_DEFAULT_MAX_TRANSACTIONS = data["TotalTransactions"].max()
FILTER_DEFAULT_MIN_SPEND = data["TotalSpend"].min()
FILTER_DEFAULT_MAX_SPEND = data["TotalSpend"].max()
FILTER_DEFAULT_MIN_TENURE = data["Tenure"].min()
FILTER_DEFAULT_MAX_TENURE = data["Tenure"].max()
FILTER_DEFAULT_GENDER = np.unique(data["Gender"]).tolist()
FILTER_DEFAULT_MARITAL_STATUS = np.unique(data["MaritalStatus"]).tolist()
FILTER_DEFAULT_EDUCATION = np.unique(data["EducationLevel"]).tolist()
FILTER_DEFAULT_CARD_TYPE = np.unique(data["CardType"]).tolist()
FILTER_DEFAULT_COUNTRY = np.unique(data["Country"]).tolist()
CATEGORICAL_CATEGORIES = {
    "Gender": FILTER_DEFAULT_GENDER,
    "MaritalStatus": FILTER_DEFAULT_MARITAL_STATUS,
    "EducationLevel": FILTER_DEFAULT_EDUCATION,
    "CardType": FILTER_DEFAULT_CARD_TYPE,
    "Country": FILTER_DEFAULT_COUNTRY,
}

# Session State
if "filter_age" not in st.session_state:
    st.session_state.filter_age = (FILTER_DEFAULT_MIN_AGE, FILTER_DEFAULT_MAX_AGE)
if "filter_gender" not in st.session_state:
    st.session_state.filter_gender = FILTER_DEFAULT_GENDER
if "filter_income" not in st.session_state:
    st.session_state.filter_income = (FILTER_DEFAULT_MIN_INCOME, FILTER_DEFAULT_MAX_INCOME)
if "filter_credit_limit" not in st.session_state:
    st.session_state.filter_credit_limit = (FILTER_DEFAULT_MIN_CREDIT_LIMIT, FILTER_DEFAULT_MAX_CREDIT_LIMIT)
if "filter_transactions" not in st.session_state:
    st.session_state.filter_transactions = (FILTER_DEFAULT_MIN_TRANSACTIONS, FILTER_DEFAULT_MAX_TRANSACTIONS)
if "filter_spend" not in st.session_state:
    st.session_state.filter_spend = (FILTER_DEFAULT_MIN_SPEND, FILTER_DEFAULT_MAX_SPEND)
if "filter_tenure" not in st.session_state:
    st.session_state.filter_tenure = (FILTER_DEFAULT_MIN_TENURE, FILTER_DEFAULT_MAX_TENURE)
if "filter_marital_status" not in st.session_state:
    st.session_state.filter_marital_status = FILTER_DEFAULT_MARITAL_STATUS
if "filter_education" not in st.session_state:
    st.session_state.filter_education = FILTER_DEFAULT_EDUCATION
if "filter_card_type" not in st.session_state:
    st.session_state.filter_card_type = FILTER_DEFAULT_CARD_TYPE
if "filter_country" not in st.session_state:
    st.session_state.filter_country = FILTER_DEFAULT_COUNTRY
if "prediction" not in st.session_state:
    st.session_state.prediction = "Click the button to predict."


# Reset Filters
def reset_filters():
    st.session_state.filter_age = (FILTER_DEFAULT_MIN_AGE, FILTER_DEFAULT_MAX_AGE)
    st.session_state.filter_gender = FILTER_DEFAULT_GENDER
    st.session_state.filter_income = (FILTER_DEFAULT_MIN_INCOME, FILTER_DEFAULT_MAX_INCOME)
    st.session_state.filter_credit_limit = (FILTER_DEFAULT_MIN_CREDIT_LIMIT, FILTER_DEFAULT_MAX_CREDIT_LIMIT)
    st.session_state.filter_transactions = (FILTER_DEFAULT_MIN_TRANSACTIONS, FILTER_DEFAULT_MAX_TRANSACTIONS)
    st.session_state.filter_spend = (FILTER_DEFAULT_MIN_SPEND, FILTER_DEFAULT_MAX_SPEND)
    st.session_state.filter_tenure = (FILTER_DEFAULT_MIN_TENURE, FILTER_DEFAULT_MAX_TENURE)
    st.session_state.filter_marital_status = FILTER_DEFAULT_MARITAL_STATUS
    st.session_state.filter_education = FILTER_DEFAULT_EDUCATION
    st.session_state.filter_card_type = FILTER_DEFAULT_CARD_TYPE
    st.session_state.filter_country = FILTER_DEFAULT_COUNTRY


# Header
st.title("Customer Churn Dashboard")
st.subheader("A Simple Dashboard for Customer Churn Analytics")
st.markdown("By: [Justin Tuazon](https://www.linkedin.com/feed/)")
st.markdown("---")

# Sidebar
st.sidebar.title("Filters")
st.sidebar.button("Reset Filters", on_click=reset_filters)

# # Filters
with st.sidebar.expander("Numerical Filters", expanded=True):
    # # # Age
    filter_min_age, filter_max_age = st.slider(
        "Age",
        FILTER_DEFAULT_MIN_AGE,
        FILTER_DEFAULT_MAX_AGE,
        (FILTER_DEFAULT_MIN_AGE, FILTER_DEFAULT_MAX_AGE),
        key="filter_age"
    )

    # # # Income
    filter_min_income, filter_max_income = st.slider(
        "Income",
        FILTER_DEFAULT_MIN_INCOME,
        FILTER_DEFAULT_MAX_INCOME,
        (FILTER_DEFAULT_MIN_INCOME, FILTER_DEFAULT_MAX_INCOME),
        key="filter_income"
    )

    # # # Credit Limit
    filter_min_credit_limit, filter_max_credit_limit = st.slider(
        "Credit Limit",
        FILTER_DEFAULT_MIN_CREDIT_LIMIT,
        FILTER_DEFAULT_MAX_CREDIT_LIMIT,
        (FILTER_DEFAULT_MIN_CREDIT_LIMIT, FILTER_DEFAULT_MAX_CREDIT_LIMIT),
        key="filter_credit_limit"
    )

    # # # Transactions
    filter_min_transactions, filter_max_transactions = st.slider(
        "Transactions",
        FILTER_DEFAULT_MIN_TRANSACTIONS,
        FILTER_DEFAULT_MAX_TRANSACTIONS,
        (FILTER_DEFAULT_MIN_TRANSACTIONS, FILTER_DEFAULT_MAX_TRANSACTIONS),
        key="filter_transactions"
    )

    # # # Spending
    filter_min_spend, filter_max_spend = st.slider(
        "Spending",
        FILTER_DEFAULT_MIN_SPEND,
        FILTER_DEFAULT_MAX_SPEND,
        (FILTER_DEFAULT_MIN_SPEND, FILTER_DEFAULT_MAX_SPEND),
        key="filter_spend"
    )

    # # # Tenure
    filter_min_tenure, filter_max_tenure = st.slider(
        "Tenure",
        FILTER_DEFAULT_MIN_TENURE,
        FILTER_DEFAULT_MAX_TENURE,
        (FILTER_DEFAULT_MIN_TENURE, FILTER_DEFAULT_MAX_TENURE),
        key="filter_tenure"
    )

with st.sidebar.expander("Categorical Filters", expanded=True):
    # # # Gender
    gender_filter = st.multiselect(
        "Gender",
        FILTER_DEFAULT_GENDER,
        FILTER_DEFAULT_GENDER,
        key="filter_gender"
    )

    # # # Marital Status
    filter_marital_status = st.multiselect(
        "Marital Status",
        FILTER_DEFAULT_MARITAL_STATUS,
        FILTER_DEFAULT_MARITAL_STATUS,
        key="filter_marital_status"
    )

    # # # Education Level
    filter_education = st.multiselect(
        "Education Level",
        FILTER_DEFAULT_EDUCATION,
        FILTER_DEFAULT_EDUCATION,
        key="filter_education"
    )

    # # # Card Type
    filter_card_type = st.multiselect(
        "Card Type",
        FILTER_DEFAULT_CARD_TYPE,
        FILTER_DEFAULT_CARD_TYPE,
        key="filter_card_type"
    )

    # # # Country
    filter_country = st.multiselect(
        "Country",
        FILTER_DEFAULT_COUNTRY,
        FILTER_DEFAULT_COUNTRY,
        key="filter_country"
    )

# Filter Data
filtered_data = data.copy()

age_min, age_max = st.session_state.filter_age
income_min, income_max = st.session_state.filter_income
credit_min, credit_max = st.session_state.filter_credit_limit
transactions_min, transactions_max = st.session_state.filter_transactions
spend_min, spend_max = st.session_state.filter_spend
tenure_min, tenure_max = st.session_state.filter_tenure

filtered_data = filtered_data[
    (filtered_data["Age"] >= age_min) & (filtered_data["Age"] <= age_max) &
    (filtered_data["Income"] >= income_min) & (filtered_data["Income"] <= income_max) &
    (filtered_data["CreditLimit"] >= credit_min) & (filtered_data["CreditLimit"] <= credit_max) &
    (filtered_data["TotalTransactions"] >= transactions_min) & (filtered_data["TotalTransactions"] <=
                                                                transactions_max) &
    (filtered_data["TotalSpend"] >= spend_min) & (filtered_data["TotalSpend"] <= spend_max) &
    (filtered_data["Tenure"] >= tenure_min) & (filtered_data["Tenure"] <= tenure_max)
]

filtered_data = filtered_data[filtered_data["Gender"].isin(st.session_state.filter_gender)]
filtered_data = filtered_data[filtered_data["MaritalStatus"].isin(st.session_state.filter_marital_status)]
filtered_data = filtered_data[filtered_data["EducationLevel"].isin(st.session_state.filter_education)]
filtered_data = filtered_data[filtered_data["CardType"].isin(st.session_state.filter_card_type)]
filtered_data = filtered_data[filtered_data["Country"].isin(st.session_state.filter_country)]

# Body
st.write("#### Overview")

overall_churn = filtered_data["AttritionFlagNum"].mean()
st.write(f"The current dataset contains data on {len(filtered_data)} customers. "
         f"The overall churn rate for the selection is about {overall_churn:.2%}. You can view the dataset below. "
         f"You can also use the churn predictor to determine wheter a specific customer is at-risk. "
         f"Lastly, there are several visualizations that compare active and churned customers.")

# Raw Data
with st.expander("Dataset Preview", expanded=False):
    st.dataframe(filtered_data, use_container_width=True)
with st.expander("Churn Predictor", expanded=False):
    st.write("##### Customer details")
    st.markdown("###### Quantiative details")
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        predict_age = st.number_input(
            "Enter age",
            min_value=FILTER_DEFAULT_MIN_AGE,
            max_value=FILTER_DEFAULT_MAX_AGE,
            value=int(data["Age"].median()),
            step=1,
            key="predict_age"
        )
    with col_2:
        predict_income = st.number_input(
            "Enter income",
            min_value=FILTER_DEFAULT_MIN_INCOME,
            max_value=FILTER_DEFAULT_MAX_INCOME,
            value=data["Income"].median().round(2),
            key="predict_income"
        )
    with col_3:
        predict_credit_limit = st.number_input(
            "Enter credit limit",
            min_value=FILTER_DEFAULT_MIN_CREDIT_LIMIT,
            max_value=FILTER_DEFAULT_MAX_CREDIT_LIMIT,
            value=data["CreditLimit"].median().round(2),
            key="predict_credit_limit"
        )
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        predict_transactions = st.number_input(
            "Enter transaction count",
            min_value=FILTER_DEFAULT_MIN_TRANSACTIONS,
            max_value=FILTER_DEFAULT_MAX_TRANSACTIONS,
            value=int(data["TotalTransactions"].median()),
            key="predict_transactions"
        )
    with col_2:
        predict_spend = st.number_input(
            "Enter total spending",
            min_value=FILTER_DEFAULT_MIN_SPEND,
            max_value=FILTER_DEFAULT_MAX_SPEND,
            value=data["TotalSpend"].median().round(2),
            key="predict_spend"
        )
    with col_3:
        predict_tenure = st.number_input(
            "Enter tenure",
            min_value=FILTER_DEFAULT_MIN_TENURE,
            max_value=FILTER_DEFAULT_MAX_TENURE,
            value=int(data["Tenure"].median()),
            key="predict_tenure"
        )

    st.markdown("###### Qualitative details")
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        predict_gender = st.selectbox(
            "Select gender",
            FILTER_DEFAULT_GENDER,
            index=0,
            key="predict_gender"
        )
    with col_2:
        predict_marital_status = st.selectbox(
            "Select marital status",
            FILTER_DEFAULT_MARITAL_STATUS,
            index=0,
            key="predict_marital_status"
        )
    with col_3:
        predict_education = st.selectbox(
            "Select education level",
            FILTER_DEFAULT_EDUCATION,
            index=0,
            key="predict_education"
        )
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        predict_card_type = st.selectbox(
            "Select card type",
            FILTER_DEFAULT_CARD_TYPE,
            index=0,
            key="predict_card_type"
        )
    with col_2:
        predict_country = st.selectbox(
            "Select country",
            FILTER_DEFAULT_COUNTRY,
            index=0,
            key="predict_country"
        )

    st.markdown("###### Prediction")
    col1, col2, col_3 = st.columns(3)
    with col1:
        st.button("Predict Attrition Status", on_click=get_prediction)
    with col2:
        st.write(st.session_state.prediction)

st.markdown("#### Table of Contents")
st.markdown("""
- [Income and Spending](#income-and-spending)
- [Credit Limit and Total Transactions](#credit-limit-and-total-transactions)
- [Age and Tenure](#age-and-tenure)
- [Card Type](#card-type)
- [Education Level](#education-level)
- [Country](#country)
- [Gender and Marital Status](#gender-and-marital-status)
""", unsafe_allow_html=True)

# Income and Spending
st.markdown("#### Income and Spending")
temp = filtered_data.groupby("AttritionFlag")[["Income", "TotalSpend"]].mean().reset_index()
temp.columns = ["Attrition Status", "Mean Income", "Mean Total Spending"]
st.dataframe(temp, use_container_width=True)

col_1, col_2 = st.columns(2)

with col_1:
    fig = px.histogram(
        filtered_data,
        x="Income",
        color="AttritionFlag",
        barmode="overlay",
        title="Income Distribution"
    )
    fig.update_layout(
        xaxis_title="Customer Income (Php)",
        yaxis_title="Number of Customers",
        legend_title="Attrition Status"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_2:
    fig = px.histogram(
        filtered_data,
        x="TotalSpend",
        color="AttritionFlag",
        barmode="overlay",
        title="Spending Distribution"
    )
    fig.update_layout(
        xaxis_title="Customer Total Spending (Php)",
        yaxis_title="Number of Customers",
        legend_title="Attrition Status"
    )
    st.plotly_chart(fig, use_container_width=True)

# Credit Limit and Transactions
st.markdown("#### Credit Limit and Total Transactions")
temp = filtered_data.groupby("AttritionFlag")[["CreditLimit", "TotalTransactions"]].mean().reset_index()
temp.columns = ["Attrition Status", "Mean Credit Limit", "Mean Total Transactions"]
st.dataframe(temp, use_container_width=True)
col_1, col_2 = st.columns(2)

with col_1:
    fig = px.histogram(
        filtered_data,
        x="CreditLimit",
        color="AttritionFlag",
        barmode="overlay",
        title="Credit Limit Distribution"
    )
    fig.update_layout(
        xaxis_title="Credit Limit (Php)",
        yaxis_title="Number of Customers",
        legend_title="Attrition Status"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_2:
    fig = px.histogram(
        filtered_data,
        x="TotalTransactions",
        color="AttritionFlag",
        barmode="overlay",
        title="Transaction Count Distribution"
    )
    fig.update_layout(
        xaxis_title="Number of Transactions",
        yaxis_title="Number of Customers",
        legend_title="Attrition Status"
    )
    st.plotly_chart(fig, use_container_width=True)

# Age and Tenure
st.markdown("#### Age and Tenure")
temp = filtered_data.groupby("AttritionFlag")[["Age", "Tenure"]].mean().reset_index()
temp.columns = ["Attrition Status", "Mean Age", "Mean Tenure"]
st.dataframe(temp, use_container_width=True)
col_1, col_2 = st.columns(2)

with col_1:
    fig = px.histogram(
        filtered_data,
        x="Age",
        color="AttritionFlag",
        barmode="overlay",
        title="Age Distribution"
    )
    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Number of Customers",
        legend_title="Attrition Status"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_2:
    fig = px.histogram(
        filtered_data,
        x="Tenure",
        color="AttritionFlag",
        barmode="overlay",
        title="Tenure Distribution"
    )
    fig.update_layout(
        xaxis_title="Tenure",
        yaxis_title="Number of Customers",
        legend_title="Attrition Status"
    )
    st.plotly_chart(fig, use_container_width=True)

# Card Type
st.markdown("#### Card Type")
temp = filtered_data.groupby(["CardType"])["AttritionFlagNum"].mean().reset_index()
temp.sort_values("AttritionFlagNum", ascending=False, inplace=True)
fig = px.bar(
    temp,
    x="CardType",
    y="AttritionFlagNum",
    title="Churn Rate Across Card Types"
)
fig.update_layout(
    xaxis_title="Card Type",
    yaxis_title="Churn Rate"
)
fig.update_layout(yaxis_tickformat=".2%")
st.plotly_chart(fig, use_container_width=True)

# Education Level
st.markdown("#### Education Level")
temp = filtered_data.groupby(["EducationLevel"])["AttritionFlagNum"].mean().reset_index()
temp.sort_values("AttritionFlagNum", ascending=False, inplace=True)
fig = px.bar(
    temp,
    x="EducationLevel",
    y="AttritionFlagNum",
    title="Churn Rate Across Card Types"
)
fig.update_layout(
    xaxis_title="Education Level",
    yaxis_title="Churn Rate"
)
fig.update_layout(yaxis_tickformat=".2%")
st.plotly_chart(fig, use_container_width=True)

# Card Type
st.markdown("#### Country")
temp = filtered_data.groupby(["Country"])["AttritionFlagNum"].mean().reset_index()
temp.sort_values("AttritionFlagNum", ascending=False, inplace=True)
fig = px.bar(
    temp,
    x="Country",
    y="AttritionFlagNum",
    title="Churn Rate Across Card Types"
)
fig.update_layout(
    xaxis_title="Country",
    yaxis_title="Churn Rate"
)
fig.update_layout(yaxis_tickformat=".2%")
st.plotly_chart(fig, use_container_width=True)

# Gender and Marital Status
st.markdown("#### Gender and Marital Status")
temp = (filtered_data.groupby(["Gender", "MaritalStatus"])["AttritionFlagNum"].mean().
        reset_index().pivot_table("AttritionFlagNum", index="Gender", columns="MaritalStatus").reset_index())
temp.columns = temp.columns.map(str)
temp.rename(columns={"Gender": "Gender / Marital Status - Churn Rate"}, inplace=True)
for col in temp.columns[1:]:
    temp[col] = (temp[col] * 100).round(2).astype(str) + "%"
st.dataframe(temp, use_container_width=True)
