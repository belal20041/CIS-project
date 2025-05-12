# pages/milestone1.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import os
import uuid

st.set_page_config(page_title="Milestone 1: Data Collection, Exploration, Preprocessing", layout="wide")
st.markdown("<h1 style='text-align: center;'>Milestone 1: Data Collection, Exploration, Preprocessing</h1>", unsafe_allow_html=True)

def load_data(file, dataset_type="train"):
    """Load dataset. Assigned to: Mohamed Samy (Standardization)"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        st.error("Unsupported format. Use CSV.")
        return None
    df = df.loc[:, ~df.columns.str.contains('^unnamed')]  # Hoda Magdy: Fix inconsistencies
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")  # Standardize names
    return df

def detect_column_types(df, date_col=None):
    """Detect column types. Assigned to: Mohamed Samy"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = [
        col for col in df.columns
        if col != date_col and (df[col].dtype in ['object', 'category', 'bool'] or df[col].nunique() / len(df) < 0.05)
    ]
    if date_col:
        numeric_cols = [col for col in numeric_cols if col != date_col]
        categorical_cols = [col for col in categorical_cols if col != date_col]
    return numeric_cols, categorical_cols

def clean_text_columns(df, columns):
    """Clean text columns and fix typos. Assigned to: Hoda Magdy"""
    df_clean = df.copy()
    def clean_text(text):
        if isinstance(text, str):
            return text.strip().capitalize()
        return text
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_text)
    corrections = {
        "fundacion de guayaquil-1": "Fundacion de guayaquil",
        "santo domingo de los tsachilas": "Santo domingo"
    }
    for col in ['description', 'locale_name']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(lambda x: corrections.get(x, x))
    return df_clean

def explore_data(df, date_col=None, numeric_cols=None, categorical_cols=None, dataset_type="train"):
    """EDA: trends, seasonality, outliers. Assigned to: Belal Khamis"""
    with st.container():
        st.markdown(f"### {dataset_type.capitalize()} Data Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape**:", df.shape)
            st.write("**Missing Values**:", df.isna().sum().to_dict())
            st.write(f"**Duplicates**: {df.duplicated().sum()}")
        with col2:
            st.write("**Data Types**:")
            st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Type', 'index': 'Column'}))
            st.write("**Unique Values per Column**:", df.nunique().to_dict())

        # Missing values visualization
        st.markdown("**Missing Values Matrix**")
        fig, ax = plt.subplots(figsize=(10, 4))
        msno.matrix(df, ax=ax)
        st.pyplot(fig)

        # Visualizations
        sns.set_style("whitegrid")
        if date_col and date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                if 'sales' in df.columns:
                    st.markdown("**Sales Trends**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.lineplot(x=df[date_col], y=df['sales'], ax=ax)
                    ax.set_title("Sales Over Time")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Sales")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            except:
                st.warning(f"Cannot process {date_col} for trends.")

        # Sales distribution by categorical columns
        for col in ['family', 'city', 'state', 'store_nbr']:
            if col in df.columns and 'sales' in df.columns:
                st.markdown(f"**Sales Distribution by {col.capitalize()}**")
                fig, ax = plt.subplots(figsize=(12, 4))
                sns.boxplot(data=df, x=col, y='sales', ax=ax)
                ax.set_title(f"Sales Distribution by {col.capitalize()}")
                ax.set_xlabel(col.capitalize())
                ax.set_ylabel("Sales")
                plt.xticks(rotation=45)
                st.pyplot(fig)

        # Sales vs. promotions
        if 'sales' in df.columns and 'onpromotion' in df.columns:
            st.markdown("**Effect of Promotions on Sales**")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df, x='onpromotion', y='sales', ax=ax)
            ax.set_title("Effect of Promotions on Sales")
            ax.set_xlabel("On Promotion (0 = No, 1 = Yes)")
            ax.set_ylabel("Sales")
            st.pyplot(fig)

        # Sales distribution
        if 'sales' in df.columns:
            st.markdown("**Sales Distribution**")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df['sales'], bins=30, kde=True, ax=ax)
            ax.set_title("Sales Distribution")
            ax.set_xlabel("Sales")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # Correlation heatmap
        if numeric_cols:
            st.markdown("**Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

        # Outliers
        if numeric_cols:
            valid_numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(df[col])]
            if valid_numeric_cols:
                st.markdown("**Outliers**")
                fig, axes = plt.subplots(1, len(valid_numeric_cols), figsize=(4*len(valid_numeric_cols), 4))
                axes = [axes] if len(valid_numeric_cols) == 1 else axes
                for i, col in enumerate(valid_numeric_cols):
                    sns.boxplot(x=df[col], ax=axes[i])
                    axes[i].set_title(f"{col}")
                plt.tight_layout()
                st.pyplot(fig)

                st.write("**Outlier Counts**:")
                for col in valid_numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    col_data = df[col].dropna()
                    if col_data.std() > 1e-6 and len(col_data) > 0:
                        z_scores = zscore(col_data)
                        outliers_zscore = df.loc[col_data.index][abs(z_scores) > 3][col]
                    else:
                        outliers_zscore = pd.Series(dtype=df[col].dtype)
                    st.write(f"{col}: IQR Outliers = {len(outliers_iqr)}, Z-score Outliers = {len(outliers_zscore)}")

def preprocess_data(df, numeric_cols=None, categorical_cols=None, date_col=None, handle_outliers='remove', scale=False, dataset_type="train"):
    """Preprocess data. Assigned to: Belal Khamis, Marwa Kotb, Mahmoud Sabry, Mohamed Samy, Hoda Magdy"""
    df_clean = df.copy()

    # Missing values. Assigned to: Belal Khamis
    for col in df_clean.columns:
        if df_clean[col].isna().sum() > 0:
            if col in numeric_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif col in categorical_cols:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            else:
                df_clean[col].fillna('Unknown', inplace=True)

    # Duplicates. Assigned to: Marwa Kotb
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()

    # Outliers. Assigned to: Mahmoud Sabry
    if numeric_cols and handle_outliers:
        if handle_outliers == 'remove':
            for col in numeric_cols:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        elif handle_outliers == 'replace':
            for col in numeric_cols:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    median_value = df_clean[col].median()
                    df_clean[col] = df_clean[col].apply(
                        lambda x: median_value if (x < lower_bound or x > upper_bound) else x
                    )

    # Time features. Assigned to: Mohamed Samy
    if date_col and date_col in df_clean.columns:
        try:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
            df_clean['year'] = df_clean[date_col].dt.year
            df_clean['month'] = df_clean[date_col].dt.month
            df_clean['day'] = df_clean[date_col].dt.day
            df_clean['dayofweek'] = df_clean[date_col].dt.weekday
            df_clean['week'] = df_clean[date_col].dt.isocalendar().week
            df_clean['is_weekend'] = df_clean['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
            df_clean['season'] = pd.cut(df_clean['month'], bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4'])
        except:
            st.warning(f"Cannot process time features for {date_col}.")

    # Standardization. Assigned to: Mohamed Samy
    if scale and numeric_cols:
        valid_numeric_cols = [col for col in numeric_cols if col in df_clean.columns and df_clean[col].std() > 1e-6]
        if valid_numeric_cols:
            scaler = StandardScaler()
            df_clean[valid_numeric_cols] = scaler.fit_transform(df_clean[valid_numeric_cols]) if dataset_type == "train" else scaler.transform(df_clean[valid_numeric_cols])

    # Text cleaning. Assigned to: Hoda Magdy
    text_cols = ["family", "city", "state", "cluster", "type_y", "locale", "locale_name", "description", "transferred"]
    df_clean = clean_text_columns(df_clean, text_cols)

    return df_clean, initial_rows - df_clean.shape[0]

def engineer_features(train, test, numeric_cols, categorical_cols, target_col='sales'):
    """Feature engineering. Assigned to: Mohamed Samy"""
    train_fe = train.copy()
    test_fe = test.copy()

    # Interaction terms
    if target_col in train_fe.columns and 'onpromotion' in train_fe.columns:
        train_fe['sales_onpromo'] = train_fe[target_col] * train_fe['onpromotion']
        if 'onpromotion' in test_fe.columns:
            test_fe['sales_onpromo'] = test_fe['onpromotion'] * 0  # Placeholder for test

    if 'onpromotion' in train_fe.columns and 'is_weekend' in train_fe.columns:
        train_fe['promo_weekend'] = train_fe['onpromotion'] * train_fe['is_weekend']
        if 'onpromotion' in test_fe.columns and 'is_weekend' in test_fe.columns:
            test_fe['promo_weekend'] = test_fe['onpromotion'] * test_fe['is_weekend']

    # Mean encoding
    for col in ['city', 'state']:
        if col in train_fe.columns and target_col in train_fe.columns:
            mean_map = train_fe.groupby(col)[target_col].mean().to_dict()
            train_fe[f'{col}_encoded'] = train_fe[col].map(mean_map)
            if col in test_fe.columns:
                test_fe[f'{col}_encoded'] = test_fe[col].map(mean_map).fillna(train_fe[target_col].mean())

    # Frequency encoding
    for col in ['family', 'locale_name']:
        if col in train_fe.columns:
            freq_map = train_fe[col].value_counts(normalize=True).to_dict()
            train_fe[f'{col}_encoded'] = train_fe[col].map(freq_map)
            if col in test_fe.columns:
                test_fe[f'{col}_encoded'] = test_fe[col].map(freq_map).fillna(0)

    # Average sales per store and month
    if 'store_nbr' in train_fe.columns and 'month' in train_fe.columns and target_col in train_fe.columns:
        train_fe['avg_sales_store_month'] = train_fe.groupby(['store_nbr', 'month'])[target_col].transform('mean')
        # Test set: Use train's mapping (approximation)
        if 'store_nbr' in test_fe.columns and 'month' in test_fe.columns:
            store_month_means = train_fe.groupby(['store_nbr', 'month'])[target_col].mean().to_dict()
            test_fe['avg_sales_store_month'] = test_fe.apply(
                lambda x: store_month_means.get((x['store_nbr'], x['month']), train_fe[target_col].mean()), axis=1
            )

    # Binary encoding
    for df in [train_fe, test_fe]:
        if 'description' in df.columns:
            df['is_holiday'] = df['description'].str.contains('Holiday|Navidad', case=False, na=False).astype(int)
        for col in ['transferred', 'is_weekend', 'is_holiday']:
            if col in df.columns:
                df[col] = df[col].astype(int)

    # Ordinal encoding
    locale_order = {'National': 2, 'Regional': 1, 'Local': 0}
    type_y_order = {'Holiday': 2, 'Event': 1, 'Bridge': 0}
    for df in [train_fe, test_fe]:
        if 'locale' in df.columns:
            df['locale_encoded'] = df['locale'].map(locale_order).fillna(0)
        if 'type_y' in df.columns:
            df['type_y_encoded'] = df['type_y'].map(type_y_order).fillna(0)

    # Binning
    if 'dcoilwtico' in train_fe.columns:
        q25 = train_fe['dcoilwtico'].quantile(0.25)
        q75 = train_fe['dcoilwtico'].quantile(0.75)
        if q25 == q75:
            bins = [-np.inf, q25, np.inf]
            labels = ['low', 'high']
        else:
            bins = [-np.inf, q25, q75, np.inf]
            labels = ['low', 'medium', 'high']
        train_fe['dcoilwtico_bin'] = pd.cut(train_fe['dcoilwtico'], bins=bins, labels=labels)
        if 'dcoilwtico' in test_fe.columns:
            test_fe['dcoilwtico_bin'] = pd.cut(test_fe['dcoilwtico'], bins=bins, labels=labels)

    return train_fe, test_fe

def main():
    st.divider()
    with st.expander("ℹ️ Project Info"):
        st.markdown("**Objective**: Collect, explore, and preprocess historical sales data for analysis.")
        st.markdown("**Tasks**:")
        st.write("- **Data Collection**: Acquire historical sales dataset with features like sales, date, promotions, holidays.")
        st.write("- **Data Exploration**: Perform EDA for trends, seasonality, missing values, outliers, and patterns.")
        st.write("- **Data Preprocessing**: Handle missing values, duplicates, inconsistencies, add time features, and scale data.")
        st.markdown("**Team**:")
        st.write("- Belal Khamis: Notebook outlines, Missing Values")
        st.write("- Marwa Kotb: Duplicates")
        st.write("- Mahmoud Sabry: Outliers")
        st.write("- Mohamed Samy: Standardization, Time Features, Feature Engineering")
        st.write("- Hoda Magdy: Typos and Inconsistencies")

    # Tabs for Train and Test
    train_tab, test_tab = st.tabs(["Train Data", "Test Data"])

    with train_tab:
        with st.container():
            train_file = st.file_uploader("Upload Train Data (from Milestone 1)", type=['csv'], key="train")
            if train_file:
                train_df = load_data(train_file, "train")
                if train_df is not None:
                    st.session_state['train_df'] = train_df
                    st.write("**Preview**:")
                    st.dataframe(train_df.head(), height=150)

                    # Configuration
                    with st.form("train_config"):
                        train_date_col = st.selectbox("Date Column", ['None'] + list(train_df.columns), index=0)
                        train_date_col = None if train_date_col == 'None' else train_date_col
                        train_target_col = st.selectbox("Target Column", train_df.columns, index=train_df.columns.tolist().index('sales') if 'sales' in train_df.columns else 0)
                        train_numeric_cols, train_categorical_cols = detect_column_types(train_df, train_date_col)
                        train_numeric_cols = st.multiselect("Numeric Columns", train_df.columns, default=train_numeric_cols)
                        train_categorical_cols = st.multiselect("Categorical Columns", train_df.columns, default=train_categorical_cols)
                        train_outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace with Median'], key="train_outliers")
                        train_outlier_method = train_outlier_method.lower() if train_outlier_method != 'None' else None
                        train_scale = st.checkbox("Scale Numerics")
                        st.form_submit_button("Apply Configuration", on_click=lambda: st.session_state.update({
                            'train_date_col': train_date_col,
                            'train_target_col': train_target_col,
                            'train_numeric_cols': train_numeric_cols,
                            'train_categorical_cols': train_categorical_cols,
                            'train_outlier_method': train_outlier_method,
                            'train_scale': train_scale
                        }))

                    # Exploration and Preprocessing
                    with st.form("train_process"):
                        st.markdown("**Process Train Data**")
                        submitted = st.form_submit_button("Explore, Preprocess & Engineer Features")

                        if submitted:
                            train_df = st.session_state['train_df']
                            train_date_col = st.session_state.get('train_date_col', None)
                            train_target_col = st.session_state.get('train_target_col', 'sales')
                            train_numeric_cols = st.session_state.get('train_numeric_cols', detect_column_types(train_df, train_date_col)[0])
                            train_categorical_cols = st.session_state.get('train_categorical_cols', detect_column_types(train_df, train_date_col)[1])
                            train_outlier_method = st.session_state.get('train_outlier_method', None)
                            train_scale = st.session_state.get('train_scale', False)

                            # Exploration
                            explore_data(train_df, train_date_col, train_numeric_cols, train_categorical_cols, "train")

                            # Preprocessing
                            processed_train, duplicates_removed = preprocess_data(
                                train_df, train_numeric_cols, train_categorical_cols, train_date_col,
                                train_outlier_method, train_scale, "train"
                            )
                            st.session_state['processed_train'] = processed_train
                            st.write(f"**Processed**: {duplicates_removed} duplicates removed, {processed_train.shape[0]} rows remain")
                            st.dataframe(processed_train.head(), height=150)

                            # Feature Engineering
                            if 'test_df' in st.session_state:
                                train_fe, _ = engineer_features(
                                    processed_train, st.session_state['processed_test'],
                                    train_numeric_cols, train_categorical_cols, train_target_col
                                )
                                st.session_state['train_fe'] = train_fe
                                st.markdown("**Feature Engineered Train Data**")
                                st.dataframe(train_fe.head(), height=150)

                                # Save
                                output_path = os.path.join("data", f"train_m1.csv")
                                train_fe.to_csv(output_path, index=False)
                                st.success(f"Saved to {output_path}")
                                os.system(f"dvc add {output_path}")
                                os.system(f"git add {output_path}.dvc")
                                os.system('git commit -m "Add processed train dataset to DVC"')

    with test_tab:
        with st.container():
            test_file = st.file_uploader("Upload Test Data (from Milestone 1)", type=['csv'], key="test")
            if test_file:
                test_df = load_data(test_file, "test")
                if test_df is not None:
                    st.session_state['test_df'] = test_df
                    st.write("**Preview**:")
                    st.dataframe(test_df.head(), height=150)

                    # Configuration
                    with st.form("test_config"):
                        test_date_col = st.selectbox("Date Column", ['None'] + list(test_df.columns), index=0)
                        test_date_col = None if test_date_col == 'None' else test_date_col
                        test_numeric_cols, test_categorical_cols = detect_column_types(test_df, test_date_col)
                        test_numeric_cols = st.multiselect("Numeric Columns", test_df.columns, default=test_numeric_cols)
                        test_categorical_cols = st.multiselect("Categorical Columns", test_df.columns, default=test_categorical_cols)
                        test_outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace with Median'], key="test_outliers")
                        test_outlier_method = test_outlier_method.lower() if test_outlier_method != 'None' else None
                        test_scale = st.checkbox("Scale Numerics")
                        st.form_submit_button("Apply Configuration", on_click=lambda: st.session_state.update({
                            'test_date_col': test_date_col,
                            'test_numeric_cols': test_numeric_cols,
                            'test_categorical_cols': test_categorical_cols,
                            'test_outlier_method': test_outlier_method,
                            'test_scale': test_scale
                        }))

                    # Exploration and Preprocessing
                    with st.form("test_process"):
                        st.markdown("**Process Test Data**")
                        submitted = st.form_submit_button("Explore, Preprocess & Engineer Features")

                        if submitted:
                            test_df = st.session_state['test_df']
                            test_date_col = st.session_state.get('test_date_col', None)
                            test_numeric_cols = st.session_state.get('test_numeric_cols', detect_column_types(test_df, test_date_col)[0])
                            test_categorical_cols = st.session_state.get('test_categorical_cols', detect_column_types(test_df, test_date_col)[1])
                            test_outlier_method = st.session_state.get('test_outlier_method', None)
                            test_scale = st.session_state.get('test_scale', False)

                            # Exploration
                            explore_data(test_df, test_date_col, test_numeric_cols, test_categorical_cols, "test")

                            # Preprocessing
                            processed_test, duplicates_removed = preprocess_data(
                                test_df, test_numeric_cols, test_categorical_cols, test_date_col,
                                test_outlier_method, test_scale, "test"
                            )
                            st.session_state['processed_test'] = processed_test
                            st.write(f"**Processed**: {duplicates_removed} duplicates removed, {processed_test.shape[0]} rows remain")
                            st.dataframe(processed_test.head(), height=150)

                            # Feature Engineering
                            if 'processed_train' in st.session_state:
                                _, test_fe = engineer_features(
                                    st.session_state['processed_train'], processed_test,
                                    st.session_state.get('train_numeric_cols', test_numeric_cols),
                                    st.session_state.get('train_categorical_cols', test_categorical_cols),
                                    st.session_state.get('train_target_col', 'sales')
                                )
                                st.session_state['test_fe'] = test_fe
                                st.markdown("**Feature Engineered Test Data**")
                                st.dataframe(test_fe.head(), height=150)

                                # Save
                                output_path = os.path.join("data", f"test_m1.csv")
                                test_fe.to_csv(output_path, index=False)
                                st.success(f"Saved to {output_path}")
                                os.system(f"dvc add {output_path}")
                                os.system(f"git add {output_path}.dvc")
                                os.system('git commit -m "Add processed test dataset to DVC"')

    st.divider()
    st.markdown("**Created with Belal Khamis, All Rights Reserved**")

if __name__ == "__main__":
    main()
