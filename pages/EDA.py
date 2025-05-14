import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="EDA", layout="wide")
st.markdown("<h1 style='text-align: center;'>EDA</h1>", unsafe_allow_html=True)

def load_data(uploaded_file=None):
    """Load dataset from uploaded file."""
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
        return df
    return None

def detect_column_types(df):
    """Detect numeric and categorical columns."""
    numeric_cols = df.select_dtypes(['int64', 'float64']).columns.tolist()
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() / len(df) < 0.05]
    return numeric_cols, categorical_cols

def explore_data(df, date_col=None, numeric_cols=None, categorical_cols=None):
    """Minimal EDA with key visualizations."""
    st.markdown("### Data Insights")
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isna().sum().to_dict())
    st.write("Duplicates:", df.duplicated().sum())

    # Missing values
    fig, ax = plt.subplots(figsize=(8, 3))
    msno.matrix(df, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    # Sales trends
    if date_col and 'sales' in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.lineplot(x=date_col, y='sales', data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    # Correlation heatmap
    if numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

def preprocess_data(df, numeric_cols, categorical_cols, date_col=None, scale=False):
    """Basic preprocessing."""
    df_clean = df.copy()

    # Missing values
    for col in df_clean.columns:
        if df_clean[col].isna().sum() > 0:
            if col in numeric_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif col in categorical_cols:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    # Duplicates
    df_clean = df_clean.drop_duplicates()

    # Time features
    if date_col:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean['year'] = df_clean[date_col].dt.year
        df_clean['month'] = df_clean[date_col].dt.month
        df_clean['is_weekend'] = df_clean[date_col].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

    # Scale
    if scale and numeric_cols:
        scaler = StandardScaler()
        df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    return df_clean

def main():
    # File upload
    train_file = st.file_uploader("Upload Train CSV", type=["csv"])

    # Load data
    train_df = load_data(train_file)
    if train_df is None:
        st.error("Please upload a CSV file.")
        return
    if train_df.empty:
        st.error("Loaded data is empty.")
        return

    st.dataframe(train_df.head())

    # Config
    date_col = st.selectbox("Date Column", ['None'] + list(train_df.columns), index=train_df.columns.tolist().index('date') if 'date' in train_df.columns else 0)
    date_col = None if date_col == 'None' else date_col
    numeric_cols, categorical_cols = detect_column_types(train_df)
    numeric_cols = st.multiselect("Numeric Columns", train_df.columns, default=numeric_cols)
    categorical_cols = st.multiselect("Categorical Columns", train_df.columns, default=categorical_cols)
    scale = st.checkbox("Scale Numerics")

    # Process
    if st.button("Process Data"):
        explore_data(train_df, date_col, numeric_cols, categorical_cols)
        processed_df = preprocess_data(train_df, numeric_cols, categorical_cols, date_col, scale)
        st.write("Processed Data")
        st.dataframe(processed_df.head())
        csv = processed_df.to_csv(index=False)
        st.download_button("Download Processed Data", csv, "train_processed.csv", "text/csv")

if __name__ == "__main__":
    main()
