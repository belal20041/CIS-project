import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Streamlit page configuration
st.set_page_config(page_title="Milestone 2: Feature Engineering", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>MILESTONE 2: Feature Engineering</h1>", unsafe_allow_html=True)
st.markdown("This milestone builds on Milestone 1 by adding advanced feature engineering. Upload your data and specify columns to proceed.")

# Function to preprocess data (replicating relevant parts of milestone1.py)
def preprocess_data(df, date_col, numeric_cols, categorical_cols):
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    
    # Convert date column to datetime
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        # Extract time-based features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['weekday'] = df[date_col].dt.weekday
        df['week'] = df[date_col].dt.isocalendar().week
        df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
        df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # Convert categorical columns to category dtype
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Ensure numeric columns are numeric
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    return df

# Function for feature engineering
def engineer_features(train, test, numeric_cols, categorical_cols, target_col):
    train = train.copy()
    test = test.copy()
    
    # Interaction terms
    if target_col in train.columns and 'onpromotion' in train.columns:
        train['sales_onpromo'] = train[target_col] * train['onpromotion']
        if 'onpromotion' in test.columns:
            test['sales_onpromo'] = test['onpromotion'] * 0  # Placeholder for test
    
    if 'onpromotion' in train.columns and 'is_weekend' in train.columns:
        train['promo_weekend'] = train['onpromotion'] * train['is_weekend']
        if 'onpromotion' in test.columns and 'is_weekend' in test.columns:
            test['promo_weekend'] = test['onpromotion'] * test['is_weekend']
    
    # Mean encoding for categorical columns
    for col in categorical_cols:
        if col in train.columns:
            mean_map = train.groupby(col)[target_col].mean().to_dict()
            train[f'{col}_encoded'] = train[col].map(mean_map)
            test[f'{col}_encoded'] = test[col].map(mean_map).fillna(train[target_col].mean())
    
    # Frequency encoding for categorical columns
    for col in categorical_cols:
        if col in train.columns:
            freq_map = train[col].value_counts(normalize=True).to_dict()
            train[f'{col}_freq'] = train[col].map(freq_map)
            test[f'{col}_freq'] = test[col].map(freq_map).fillna(0)
    
    # Standardize numeric columns
    scaler = StandardScaler()
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])
    
    return train, test

# Function to plot feature distributions
def plot_features(df, columns, target_col):
    for col in columns:
        st.subheader(f"Distribution of {col}")
        fig, ax = plt.subplots(figsize=(10, 6))
        if col == target_col:
            sns.histplot(df[col], bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
        else:
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot of {col}")
        plt.tight_layout()
        st.pyplot(fig)

# Main Streamlit app
def main():
    # File upload
    st.subheader("Upload Data")
    train_file = st.file_uploader("Upload Train CSV", type=["csv"])
    test_file = st.file_uploader("Upload Test CSV", type=["csv"])
    
    if train_file and test_file:
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        
        # Column selection
        st.subheader("Specify Columns")
        all_cols = train.columns.tolist()
        date_col = st.selectbox("Select Date Column", ["None"] + all_cols, index=0)
        target_col = st.selectbox("Select Target Column", all_cols)
        numeric_cols = st.multiselect("Select Numeric Columns", all_cols)
        categorical_cols = st.multiselect("Select Categorical Columns", all_cols)
        
        if st.button("Process Data"):
            # Preprocess data
            date_col = None if date_col == "None" else date_col
            train = preprocess_data(train, date_col, numeric_cols, categorical_cols)
            test = preprocess_data(test, date_col, numeric_cols, categorical_cols)
            
            # Feature engineering
            train_m2, test_m2 = engineer_features(train, test, numeric_cols, categorical_cols, target_col)
            
            # Display results
            st.subheader("Processed Train Data")
            st.write(train_m2.head())
            st.subheader("Processed Test Data")
            st.write(test_m2.head())
            
            # Plot features
            plot_cols = numeric_cols + [f"{col}_encoded" for col in categorical_cols]
            plot_features(train_m2, plot_cols, target_col)
            
            # Save processed data
            train_m2.to_csv("train_m2.csv", index=False)
            test_m2.to_csv("test_m2.csv", index=False)
            st.success("Processed data saved as 'train_m2.csv' and 'test_m2.csv'")

if __name__ == "__main__":
    main() 
