import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import periodogram
from io import BytesIO
import os
import tempfile

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Retail Sales Time Series Analysis</h1>", unsafe_allow_html=True)

TRAIN_END = "2017-07-15"
VAL_END = "2017-08-15"

def detect_column_types(df, date_col):
    numeric_cols = df.select_dtypes(['int64', 'float64']).columns.tolist()
    if date_col in numeric_cols:
        numeric_cols.remove(date_col)
    categorical_cols = [col for col in df.columns if col != date_col and 
                       (df[col].dtype in ['object', 'category', 'bool'] or df[col].nunique() / len(df) < 0.05)]
    return numeric_cols, categorical_cols

def load_data(file, date_col, target_col):
    if hasattr(file, 'seek'):
        file.seek(0)
    df = pd.read_csv(file)
    df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y')
    df[['store_nbr', 'onpromotion']] = df[['store_nbr', 'onpromotion']].astype('int32')
    if target_col and target_col in df.columns:
        df[target_col] = df[target_col].astype('float32')
    df.dropna(subset=[date_col], inplace=True)
    return df

def prepare_data(train, test, date_col, target_col):
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test]).sort_values(['store_nbr', 'family', date_col])
    agg_dict = {target_col: 'sum', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first'}  # Changed to 'sum' for sales aggregation
    combined = combined.groupby(['store_nbr', 'family', date_col]).agg(agg_dict).reset_index()
    combined = combined.astype({'store_nbr': 'int32', 'family': 'category', date_col: 'datetime64[ns]', 
                                target_col: 'float32', 'onpromotion': 'int32', 'is_train': 'int8'})
    return combined

def fill_missing(combined, target_col):
    grouped = combined.groupby(['store_nbr', 'family'])
    processed_groups = []
    for (store_nbr, family), group in grouped:
        group[target_col] = group[target_col].ffill().fillna(0).astype('float32')
        group['onpromotion'] = group['onpromotion'].fillna(0).astype('int32')
        processed_groups.append(group)
    return pd.concat(processed_groups)

def add_features(combined, date_col, target_col):
    combined['day'] = combined[date_col].dt.day.astype('int8')
    combined['dow'] = combined[date_col].dt.dayofweek.astype('int8')
    combined['month'] = combined[date_col].dt.month.astype('int8')
    combined['year'] = combined[date_col].dt.year.astype('int16')
    combined['sin_month'] = np.sin(2 * np.pi * combined['month'] / 12).astype('float32')
    lags = [7, 14]
    for lag in lags:
        combined[f'lag_{lag}'] = combined.groupby(['store_nbr', 'family'])[target_col].shift(lag).astype('float32')
    combined['roll_mean_7'] = combined.groupby(['store_nbr', 'family'])[target_col].shift(1).rolling(7, min_periods=1).mean().astype('float32')
    combined['store_nbr_encoded'] = LabelEncoder().fit_transform(combined['store_nbr']).astype('int8')
    combined['family_encoded'] = LabelEncoder().fit_transform(combined['family']).astype('int8')
    feature_cols = ['onpromotion', 'day', 'dow', 'month', 'year', 'sin_month', 'store_nbr_encoded', 'family_encoded', 'lag_7', 'lag_14', 'roll_mean_7']
    combined[feature_cols] = StandardScaler().fit_transform(combined[feature_cols].fillna(0)).astype('float32')
    return combined

def split_data(combined, date_col, target_col):
    train = combined[combined['is_train'] == 1]
    test = combined[combined['is_train'] == 0].drop([target_col], axis=1)
    train_set = train[train[date_col] <= TRAIN_END]
    val_set = train[(train[date_col] > TRAIN_END) & (train[date_col] <= VAL_END)]
    return train_set, val_set, test

def get_download_file(df, filename):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue(), 'text/csv'

def explore_data(df, date_col, target_col, numeric_cols, categorical_cols, dataset_type):
    temp_dir = tempfile.gettempdir()
    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape:", df.shape)
        st.write("Missing:", df.isna().sum().to_dict())
        st.write("Duplicates:", df.duplicated().sum())
    with col2:
        st.write("Types:", df.dtypes.to_dict())
        st.write("Uniques:", df.nunique().to_dict())

    st.subheader(f"{dataset_type.capitalize()} Sales Insights")

    # 1. Missingness Matrix
    fig, ax = plt.subplots(figsize=(12, 5))
    msno.matrix(df, ax=ax)
    ax.set_title("Data Missingness Overview")
    plt.savefig(os.path.join(temp_dir, f"{dataset_type}_missing.png"))
    st.pyplot(fig)
    plt.close(fig)

    if target_col and target_col in df.columns:
        # 2. Total Sales Trends Over Time
        sales_by_date = df.groupby(date_col)['sales'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sales_by_date[date_col], sales_by_date['sales'], color='blue', label='Total Sales')
        holidays = df[df['type_y'] == 'Holiday'][date_col].unique()
        for holiday in holidays:
            ax.axvline(holiday, color='red', linestyle='--', alpha=0.5, label='Holiday' if holiday == holidays[0] else "")
        ax.grid(True, alpha=0.3)
        ax.set_title("Total Sales Trends with Holiday Impact")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales")
        plt.xticks(rotation=45)
        ax.legend()
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_trends.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 3. Sales by Product Family
        sales_by_family = df.groupby('family')['sales'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        sales_by_family.plot(kind='bar', ax=ax, color='skyblue')
        ax.grid(True, alpha=0.3)
        ax.set_title("Total Sales by Product Category")
        ax.set_xlabel("Product Family")
        ax.set_ylabel("Total Sales")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_family_sales.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 4. Sales by Store and City
        sales_by_store = df.groupby(['store_nbr', 'city'])['sales'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=sales_by_store, x='store_nbr', y='sales', hue='city', ax=ax, palette='muted')
        ax.grid(True, alpha=0.3)
        ax.set_title("Sales by Store Across Cities")
        ax.set_xlabel("Store Number")
        ax.set_ylabel("Total Sales")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_store_city_sales.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 5. Impact of Promotions on Sales
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df, x='onpromotion', y='sales', hue='family', ax=ax, palette='muted')
        ax.grid(True, alpha=0.3)
        ax.set_title("Sales Distribution by Promotion (by Family)")
        ax.set_xlabel("On Promotion (0 = No, 1 = Yes)")
        ax.set_ylabel("Sales")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_promo_sales.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 6. Sales During Holidays vs. Non-Holidays
        df['is_holiday'] = df['type_y'].apply(lambda x: 1 if x == 'Holiday' else 0)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df, x='is_holiday', y='sales', hue='family', ax=ax, palette='muted')
        ax.grid(True, alpha=0.3)
        ax.set_title("Sales: Holidays vs. Non-Holidays (by Family)")
        ax.set_xlabel("Holiday (0 = No, 1 = Yes)")
        ax.set_ylabel("Sales")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_holiday_sales.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 7. Sales vs. Oil Price
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.scatterplot(data=df, x='dcoilwtico', y='sales', ax=ax, color='blue', alpha=0.5)
        sns.regplot(data=df, x='dcoilwtico', y='sales', ax=ax, scatter=False, color='red')
        ax.grid(True, alpha=0.3)
        ax.set_title("Sales vs. Oil Price with Trend")
        ax.set_xlabel("Oil Price (dcoilwtico)")
        ax.set_ylabel("Sales")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_oil_sales.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 8. Monthly Sales Seasonality
        sales_by_month = df.groupby(['month', 'family'])['sales'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=sales_by_month, x='month', y='sales', hue='family', ax=ax, palette='muted')
        ax.grid(True, alpha=0.3)
        ax.set_title("Average Monthly Sales by Family")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Sales")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_monthly_sales.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 9. Heatmap: Sales Across Stores and Families
        pivot_data = df.pivot_table(values='sales', index='store_nbr', columns='family', aggfunc='sum', fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=False, cmap='YlOrRd', norm=plt.Normalize(), ax=ax)
        ax.set_title("Sales Heatmap: Stores vs. Product Families")
        ax.set_xlabel("Product Family")
        ax.set_ylabel("Store Number")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_sales_heatmap.png"))
        st.pyplot(fig)
        plt.close(fig)

def main():
    train_tab, test_tab = st.tabs(["Train", "Test"])

    with train_tab:
        train_file = st.file_uploader("Upload Train Data (.csv)", ['csv'], key="train")
        if train_file:
            if 'train_file' in st.session_state and st.session_state['train_file'] != train_file:
                for key in ['train_date_col', 'train_target_col', 'train_numeric_cols', 'train_categorical_cols', 
                           'train_outlier_method', 'train_scale', 'train_configured', 'train_df']:
                    st.session_state.pop(key, None)
            with st.form("train_config"):
                train = load_data(train_file, 'date', None)
                st.dataframe(train.head(), height=100)
                date_col = st.selectbox("Select Date Column", train.columns, index=train.columns.tolist().index('date') if 'date' in train.columns else 0, key="train_date")
                target_col = st.selectbox("Select Target Column (e.g., sales)", train.columns, index=train.columns.tolist().index('sales') if 'sales' in train.columns else 0, key="train_target")
                numeric_cols, categorical_cols = detect_column_types(train, date_col)
                numeric_cols = st.multiselect("Numeric Columns", train.columns, default=['id', 'store_nbr', 'onpromotion', 'dcoilwtico', 'transactions'] if all(col in train.columns for col in ['store_nbr', 'onpromotion', 'dcoilwtico']) else numeric_cols, key="train_numeric")
                categorical_cols = st.multiselect("Categorical Columns", train.columns, default=['family', 'city', 'state', 'type_x', 'type_y', 'locale', 'locale_name', 'description', 'transferred'] if all(col in train.columns for col in ['family', 'city']) else categorical_cols, key="train_categorical")
                outlier_method = st.selectbox("Handle Outliers", ['None', 'Remove', 'Replace'], index=2, key="train_outlier")
                outlier_method = outlier_method.lower() if outlier_method != 'None' else None
                scale = st.checkbox("Apply Scaling", key="train_scale")
                st.form_submit_button("Apply Configuration", on_click=lambda: st.session_state.update({
                    'train_date_col': date_col, 'train_target_col': target_col, 'train_numeric_cols': numeric_cols,
                    'train_categorical_cols': categorical_cols, 'train_outlier_method': outlier_method, 'train_scale': scale,
                    'train_configured': True, 'train_file': train_file
                }))

            if 'train_configured' in st.session_state and st.session_state['train_configured'] and 'train_file' in st.session_state:
                with st.form("train_process"):
                    if st.form_submit_button("Generate Insights"):
                        train = load_data(st.session_state['train_file'], st.session_state['train_date_col'], st.session_state['train_target_col'])
                        st.session_state['train_df'] = train
                        explore_data(train, st.session_state['train_date_col'], st.session_state['train_target_col'], 
                                     st.session_state['train_numeric_cols'], st.session_state['train_categorical_cols'], "train")
                        st.dataframe(train.head(), height=100)
                csv_data, mime = get_download_file(train, "train_processed.csv")
                st.download_button("Download Processed Train Data", csv_data, "train_processed.csv", mime, key="train_download")

    with test_tab:
        test_file = st.file_uploader("Upload Test Data (.csv)", ['csv'], key="test")
        if test_file:
            if 'test_file' in st.session_state and st.session_state['test_file'] != test_file:
                for key in ['test_date_col', 'test_numeric_cols', 'test_categorical_cols', 
                           'test_outlier_method', 'test_scale', 'test_configured', 'test_df']:
                    st.session_state.pop(key, None)
            with st.form("test_config"):
                test = load_data(test_file, 'date', None)
                st.dataframe(test.head(), height=100)
                date_col = st.selectbox("Select Date Column", test.columns, index=test.columns.tolist().index('date') if 'date' in test.columns else 0, key="test_date")
                numeric_cols, categorical_cols = detect_column_types(test, date_col)
                numeric_cols = st.multiselect("Numeric Columns", test.columns, default=['id', 'store_nbr', 'onpromotion', 'dcoilwtico', 'transactions'] if all(col in test.columns for col in ['store_nbr', 'onpromotion', 'dcoilwtico']) else numeric_cols, key="test_numeric")
                categorical_cols = st.multiselect("Categorical Columns", test.columns, default=['family', 'city', 'state', 'type_x', 'type_y', 'locale', 'locale_name', 'description', 'transferred'] if all(col in test.columns for col in ['family', 'city']) else categorical_cols, key="test_categorical")
                outlier_method = st.selectbox("Handle Outliers", ['None', 'Remove', 'Replace'], index=2, key="test_outlier")
                outlier_method = outlier_method.lower() if outlier_method != 'None' else None
                scale = st.checkbox("Apply Scaling", key="test_scale")
                st.form_submit_button("Apply Configuration", on_click=lambda: st.session_state.update({
                    'test_date_col': date_col, 'test_numeric_cols': numeric_cols, 'test_categorical_cols': categorical_cols,
                    'test_outlier_method': outlier_method, 'test_scale': scale, 'test_configured': True, 'test_file': test_file
                }))

            if 'test_configured' in st.session_state and st.session_state['test_configured'] and 'test_file' in st.session_state:
                with st.form("test_process"):
                    if st.form_submit_button("Generate Insights"):
                        test = load_data(st.session_state['test_file'], st.session_state['test_date_col'], None)
                        st.session_state['test_df'] = test
                        explore_data(test, st.session_state['test_date_col'], st.session_state.get('train_target_col', None), 
                                     st.session_state['test_numeric_cols'], st.session_state['test_categorical_cols'], "test")
                        st.dataframe(test.head(), height=100)
                csv_data, mime = get_download_file(test, "test_processed.csv")
                st.download_button("Download Processed Test Data", csv_data, "test_processed.csv", mime, key="test_download")

    if 'train_df' in st.session_state and 'test_df' in st.session_state:
        with st.form("feature_engineering"):
            if st.form_submit_button("Generate Features"):
                combined = prepare_data(st.session_state['train_df'], st.session_state['test_df'], 
                                       st.session_state['train_date_col'], st.session_state['train_target_col'])
                combined = fill_missing(combined, st.session_state['train_target_col'])
                combined = add_features(combined, st.session_state['train_date_col'], st.session_state['train_target_col'])
                train_set, val_set, test = split_data(combined, st.session_state['train_date_col'], st.session_state['train_target_col'])
                st.session_state['train_set'] = train_set
                st.session_state['val_set'] = val_set
                st.session_state['test_set'] = test
                st.dataframe(train_set.head(), height=100)
                st.dataframe(val_set.head(), height=100)
                st.dataframe(test.head(), height=100)
                csv_data, mime = get_download_file(train_set, "train_features.csv")
                st.download_button("Download Train Features", csv_data, "train_features.csv", mime, key="train_fe_download")
                csv_data, mime = get_download_file(val_set, "val_features.csv")
                st.download_button("Download Validation Features", csv_data, "val_features.csv", mime, key="val_fe_download")
                csv_data, mime = get_download_file(test, "test_features.csv")
                st.download_button("Download Test Features", csv_data, "test_features.csv", mime, key="test_fe_download")

    st.markdown("**Developed by Belal Khamis, Marwa Kotb, Mahmoud Sabry, Mohamed Samy, Hoda Magdy**")

if __name__ == "__main__":
    main()
