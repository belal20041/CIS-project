import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import requests
from io import StringIO
import os

st.set_page_config(page_title="Milestone 1: Data Analysis", layout="wide")
st.markdown("<h1 style='text-align: center;'>EDA</h1>", unsafe_allow_html=True)

def load_data(dataset_type="train"):
    """Load dataset from GitHub."""
    try:
        url = f"https://raw.githubusercontent.com/CIS-project/data/main/processed_{dataset_type}_{dataset_type.capitalize()}.csv"
        df = pd.read_csv(StringIO(requests.get(url).text))
        df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)].copy()
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
        return df
    except Exception as e:
        st.error(f"Failed to load {dataset_type} data: {e}")
        return None

def detect_column_types(df, date_col=None):
    """Detect numeric and categorical columns."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [col for col in df.columns if col != date_col and 
                       (df[col].dtype in ['object', 'category', 'bool'] or df[col].nunique() / len(df) < 0.05)]
    if date_col and date_col in numeric_cols:
        numeric_cols.remove(date_col)
    return numeric_cols, categorical_cols

def clean_text_columns(df, columns):
    """Clean text columns and fix typos."""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(lambda x: x.strip().capitalize() if isinstance(x, str) else x)
    corrections = {
        "fundacion de guayaquil-1": "Fundacion de guayaquil",
        "santo domingo de los tsachilas": "Santo domingo"
    }
    for col in ['description', 'locale_name']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(corrections)
    return df_clean

def explore_data(df, date_col=None, numeric_cols=None, categorical_cols=None, dataset_type="train"):
    """Perform EDA with key visualizations."""
    st.markdown(f"### {dataset_type.capitalize()} Data Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape**:", df.shape)
        st.write("**Missing Values**:", df.isna().sum().to_dict())
        st.write(f"**Duplicates**: {df.duplicated().sum()}")
    with col2:
        st.write("**Data Types**:", df.dtypes.to_dict())
        st.write("**Unique Values**:", df.nunique().to_dict())

    # Missing values visualization
    st.markdown("**Missing Values Matrix**")
    fig, ax = plt.subplots(figsize=(8, 3))
    msno.matrix(df, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    # Sales trends
    if date_col and 'sales' in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        st.markdown("**Sales Trends**")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.lineplot(x=df[date_col], y='sales', ax=ax)
        ax.set_title("Sales Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    # Sales distribution by categories
    for col in ['family', 'city', 'store_nbr']:
        if col in df.columns and 'sales' in df.columns:
            st.markdown(f"**Sales by {col.capitalize()}**")
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.boxplot(data=df, x=col, y='sales', ax=ax)
            ax.set_title(f"Sales by {col.capitalize()}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

    # Sales distribution and correlations
    if 'sales' in df.columns:
        st.markdown("**Sales Distribution**")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(df['sales'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    if numeric_cols:
        st.markdown("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # Outliers
    if numeric_cols:
        st.markdown("**Outliers**")
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers_iqr = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
            z_scores = zscore(df[col].dropna())
            outliers_zscore = df.iloc[df[col].dropna().index][abs(z_scores) > 3][col]
            st.write(f"{col}: IQR Outliers = {len(outliers_iqr)}, Z-score Outliers = {len(outliers_zscore)}")

def preprocess_data(df, numeric_cols, categorical_cols, date_col=None, handle_outliers=None, scale=False):
    """Preprocess data: handle missing values, duplicates, outliers, and time features."""
    df_clean = df.copy()

    # Missing values
    for col in df_clean.columns:
        if df_clean[col].isna().sum() > 0:
            if col in numeric_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif col in categorical_cols:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            else:
                df_clean[col].fillna('Unknown', inplace=True)

    # Duplicates
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()

    # Outliers
    if handle_outliers and numeric_cols:
        for col in numeric_cols:
            Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            if handle_outliers == 'remove':
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
            elif handle_outliers == 'replace':
                median_value = df_clean[col].median()
                df_clean[col] = df_clean[col].apply(lambda x: median_value if x < lower or x > upper else x)

    # Time features
    if date_col and date_col in df_clean.columns:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean['year'] = df_clean[date_col].dt.year
        df_clean['month'] = df_clean[date_col].dt.month
        df_clean['day'] = df_clean[date_col].dt.day
        df_clean['weekday'] = df_clean[date_col].dt.weekday
        df_clean['week'] = df_clean[date_col].dt.isocalendar().week
        df_clean['is_weekend'] = df_clean[date_col].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
        df_clean['season'] = pd.cut(df_clean['month'], bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # One-hot encode categorical columns
    if categorical_cols:
        df_clean = pd.get_dummies(df_clean, columns=categorical_cols, dummy_na=True)

    # Scale numeric columns
    if scale and numeric_cols:
        scaler = StandardScaler()
        df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    # Clean text columns
    text_cols = ['family', 'city', 'state', 'cluster', 'type_y', 'locale', 'locale_name', 'description', 'transferred']
    df_clean = clean_text_columns(df_clean, text_cols)

    return df_clean, initial_rows - df_clean.shape[0]

def engineer_features(df, numeric_cols, categorical_cols, date_col=None):
    """Engineer features for the dataset."""
    df_fe = df.copy()

    # Interaction terms
    if 'onpromotion' in df_fe and 'is_weekend' in df_fe:
        df_fe['promo_weekend'] = df_fe['onpromotion'] * df_fe['is_weekend']
    if 'description' in df_fe:
        df_fe['is_holiday'] = df_fe['description'].str.contains('Holiday|Navidad', case=False, na=False).astype(int)

    # Mean encoding for categorical columns
    for col in ['city', 'state']:
        if col in df_fe and 'sales' in df_fe:
            mean_map = df_fe.groupby(col)['sales'].mean().to_dict()
            df_fe[f'{col}_encoded'] = df_fe[col].map(mean_map)

    # Frequency encoding
    for col in ['family', 'locale_name']:
        if col in df_fe:
            freq_map = df_fe[col].value_counts(normalize=True).to_dict()
            df_fe[f'{col}_encoded'] = df_fe[col].map(freq_map)

    # Ordinal encoding
    locale_order = {'National': 2, 'Regional': 1, 'Local': 0}
    type_y_order = {'Holiday': 2, 'Event': 1, 'Bridge': 0}
    if 'locale' in df_fe:
        df_fe['locale_encoded'] = df_fe['locale'].map(locale_order).fillna(0)
    if 'type_y' in df_fe:
        df_fe['type_y_encoded'] = df_fe['type_y'].map(type_y_order).fillna(0)

    # Binning
    if 'dcoilwtico' in df_fe:
        bins = [-np.inf, df_fe['dcoilwtico'].quantile(0.25), df_fe['dcoilwtico'].quantile(0.75), np.inf]
        labels = ['low', 'medium', 'high']
        df_fe['dcoilwtico_bin'] = pd.cut(df_fe['dcoilwtico'], bins=bins, labels=labels)

    return df_fe

def main():
    with st.expander("ℹ️ Project Info"):
        st.markdown("""
        **Objective**: Analyze historical sales data for modeling.
        **Tasks**:
        - Collect: Historical sales with features (sales, date, promotions, holidays).
        - Explore: Trends, seasonality, missing values, outliers.
        - Preprocess: Handle missing values, duplicates, add time features, scale data.
        **Team**:
        - Belal Khamis: EDA, Missing Values
        - Marwa Kotb: Duplicates
        - Mahmoud Sabry: Outliers
        - Mohamed Samy: Standardization, Time Features, Feature Engineering
        - Hoda Magdy: Text Cleaning
        """)

    # Load data
    train_df = load_data("train")
    test_df = load_data("test")
    
    # Check if data loaded successfully
    if train_df is None or test_df is None:
        st.error("Data loading failed.")
        return
    if train_df.empty or test_df.empty:
        st.error("Loaded data is empty.")
        return

    st.session_state['train_df'] = train_df
    st.session_state['test_df'] = test_df

    # Tabs
    train_tab, test_tab = st.tabs(["Train Data", "Test Data"])

    for tab, dataset_type, df in [(train_tab, "train", train_df), (test_tab, "test", test_df)]:
        with tab:
            st.write(f"**{dataset_type.capitalize()} Data Preview**")
            st.dataframe(df.head(), height=150)

            # Configuration
            with st.form(f"{dataset_type}_config"):
                date_col = st.selectbox("Date Column", ['None'] + list(df.columns), 
                                       index=df.columns.tolist().index('date') if 'date' in df.columns else 0)
                date_col = None if date_col == 'None' else date_col
                numeric_cols, categorical_cols = detect_column_types(df, date_col)
                numeric_cols = st.multiselect("Numeric Columns", df.columns, default=numeric_cols)
                categorical_cols = st.multiselect("Categorical Columns", df.columns, default=categorical_cols)
                outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace'], key=f"{dataset_type}_outliers")
                outlier_method = outlier_method.lower() if outlier_method != 'None' else None
                scale = st.checkbox("Scale Numerics")
                st.form_submit_button("Apply", on_click=lambda: st.session_state.update({
                    f'{dataset_type}_date_col': date_col,
                    f'{dataset_type}_numeric_cols': numeric_cols,
                    f'{dataset_type}_categorical_cols': categorical_cols,
                    f'{dataset_type}_outlier_method': outlier_method,
                    f'{dataset_type}_scale': scale
                }))

            # Process
            with st.form(f"{dataset_type}_process"):
                st.markdown(f"**Process {dataset_type.capitalize()} Data**")
                if st.form_submit_button("Explore & Preprocess"):
                    date_col = st.session_state.get(f'{dataset_type}_date_col', None)
                    numeric_cols = st.session_state.get(f'{dataset_type}_numeric_cols', detect_column_types(df, date_col)[0])
                    categorical_cols = st.session_state.get(f'{dataset_type}_categorical_cols', detect_column_types(df, date_col)[1])
                    outlier_method = st.session_state.get(f'{dataset_type}_outlier_method', None)
                    scale = st.session_state.get(f'{dataset_type}_scale', False)

                    # Exploration
                    explore_data(df, date_col, numeric_cols, categorical_cols, dataset_type)

                    # Preprocessing
                    processed_df, duplicates_removed = preprocess_data(df, numeric_cols, categorical_cols, date_col, outlier_method, scale)
                    st.session_state[f'processed_{dataset_type}'] = processed_df
                    st.write(f"**Processed**: {duplicates_removed} duplicates removed, {processed_df.shape[0]} rows remain")
                    st.dataframe(processed_df.head(), height=150)

                    # Feature Engineering
                    fe_df = engineer_features(processed_df, numeric_cols, categorical_cols, date_col)
                    st.session_state[f'{dataset_type}_fe'] = fe_df
                    st.markdown(f"**Feature Engineered {dataset_type.capitalize()} Data**")
                    st.dataframe(fe_df.head(), height=150)

                    # Save
                    os.makedirs("data", exist_ok=True)
                    output_path = os.path.join("data", f"{dataset_type}_m1.csv")
                    fe_df.to_csv(output_path, index=False)
                    st.success(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
