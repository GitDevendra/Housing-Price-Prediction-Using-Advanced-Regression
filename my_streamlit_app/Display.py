import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load datasets with caching
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

# File paths
train_file = "train.csv"
test_file = "test.csv"
submission_file = "submission.csv"

# Load datasets
df_train = load_data(train_file)
df_test = load_data(test_file)
df_submission = load_data(submission_file)

# Streamlit App Title
st.title("🏡 Housing Data Dashboard")

# Show Data Preview
st.subheader("📋 Raw Training Data")
st.dataframe(df_train,height=350)

# Basic Statistics
st.subheader("📊 Basic Data Insights")
st.write(df_train.describe())

# Visualization: Correlation Heatmap
st.subheader("🔍 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(20, 12))
sns.heatmap(df_train.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Feature Distribution Visualization
st.subheader("📈 Feature Distribution")
feature = st.selectbox("Select a feature to visualize", df_train.columns[1:])
fig, ax = plt.subplots()
sns.histplot(df_train[feature], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Scatter Plot for Price vs. Selected Feature
st.subheader("📉 Price vs Selected Feature")
x_feature = st.selectbox("Choose X-axis feature", df_train.columns, index=list(df_train.columns).index("SalePrice") - 1)
fig, ax = plt.subplots()
sns.scatterplot(x=df_train[x_feature], y=df_train["SalePrice"], alpha=0.6, ax=ax)
st.pyplot(fig)

# Skewness Handling
st.subheader("🏠 Initial Sale Price Distribution")
fig, ax = plt.subplots()
sns.histplot(df_train['SalePrice'], kde=True, ax=ax)
st.pyplot(fig)
st.write("Data is skewed; applying log transformation...")

df_train['SalePrice'] = df_train['SalePrice'].apply(lambda x: math.log(1 + x))

st.subheader("📊 Transformed Sale Price Distribution")
fig, ax = plt.subplots()
sns.histplot(df_train['SalePrice'], kde=True, ax=ax)
st.pyplot(fig)

# Display Test Dataset
st.subheader("📋 Test Data Preview")
st.dataframe(df_test, height=350)

# Sales Price Predictions
st.header("📈 Predicted Sales Price for Each House")
st.subheader("🔍 Search for a Sales Price by ID")
search_id = st.text_input("Enter an ID to search for its Sales Price:")

if search_id:
    try:
        search_id = int(search_id)
        result = df_submission[df_submission["Id"] == search_id]
        if not result.empty:
            sales_price = result["SalePrice"].values[0]
            st.success(f"💰 The predicted Sales Price for ID **{search_id}** is **${sales_price:,.2f}**")
        else:
            st.warning("⚠️ No matching ID found in the dataset.")
    except ValueError:
        st.error("❌ Please enter a valid numeric ID.")

st.write("👨‍💻 Made by Devendra Singh")