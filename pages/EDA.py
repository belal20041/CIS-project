import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import periodogram
from io import BytesIO
import os
import tempfile

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Time Series Analysis</h1>", unsafe_allow_html=True)

def load_data(file, dataset_type):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_parquet(file)
    df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

def detect_column_types(df, date_col):
    numeric_cols = df.select_dtypes(['int64', 'float64']).columns.tolist()
    if date_col in numeric_cols:
        numeric_cols.remove(date_col)
    categorical_cols = [col for col in df.columns if col != date_col and 
                       (df[col].dtype in ['object', 'category', 'bool'] or df[col].nunique() / len(df) < 0.05)]
    return numeric_cols, categorical_cols

def clean_text_columns(df):
    df_clean = df.copy()
    for col in ['family', 'city', 'state', 'cluster', 'type_y', 'locale', 'locale_name', 'description', 'transferred']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(lambda x: x.strip().capitalize() if isinstance(x, str) else x)
    corrections = {"fundacion de guayaquil-1": "Fundacion de guayaquil", "santo domingo de los tsachilas": "Santo domingo"}
    for col in ['description', 'locale_name']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(corrections)
    return df_clean

def get_download_file(df, filename):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue(), 'text/csv'

def explore_data(df, date_col, numeric_cols, categorical_cols, dataset_type):
    temp_dir = tempfile.gettempdir()
    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape:", df.shape)
        st.write("Missing:", df.isna().sum().to_dict())
        st.write("Duplicates:", df.duplicated().sum())
    with col2:
        st.write("Types:", df.dtypes.to_dict())
        st.write("Uniques:", df.nunique().to_dict())

    with st.expander("Visualizations"):
        fig, ax = plt.subplots(figsize=(8, 3))
        msno.matrix(df, ax=ax)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_missing.png"))
        st.pyplot(fig)
        plt.close(fig)

        if date_col and 'sales' in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)

            fig, ax = plt.subplots(figsize=(8, 3))
            sns.lineplot(x=date_col, y='sales', data=df, ax=ax)
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_trends.png"))
            st.pyplot(fig)
            plt.close(fig)

            df_weekly = df.set_index(date_col)['sales'].resample('W').sum().reset_index()
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.lineplot(x=date_col, y='sales', data=df_weekly, ax=ax)
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_weekly.png"))
            st.pyplot(fig)
            plt.close(fig)

            df['month'] = df[date_col].dt.month
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.boxplot(x='month', y='sales', data=df, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_seasonal.png"))
            st.pyplot(fig)
            plt.close(fig)

            rolling = df.set_index(date_col)['sales'].rolling(window=30).agg(['mean', 'std']).dropna()
            fig, ax = plt.subplots(figsize=(8, 3))
            rolling['mean'].plot(ax=ax, label='Mean')
            rolling['std'].plot(ax=ax, label='Std', alpha=0.5)
            plt.legend()
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_rolling.png"))
            st.pyplot(fig)
            plt.close(fig)

            df_ts = df.set_index(date_col)['sales'].resample('M').sum()
            decomp = seasonal_decompose(df_ts, model='additive', period=12)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
            decomp.trend.plot(ax=ax1)
            decomp.seasonal.plot(ax=ax2)
            decomp.resid.plot(ax=ax3)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_decomp.png"))
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 3))
            plot_acf(df['sales'].dropna(), lags=30, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_acf.png"))
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 3))
            plot_pacf(df['sales'].dropna(), lags=30, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_pacf.png"))
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 3))
            pd.plotting.lag_plot(df['sales'], lag=1, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_lag.png"))
            st.pyplot(fig)
            plt.close(fig)

            freq, psd = periodogram(df['sales'].dropna())
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(freq, psd)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_periodogram.png"))
            st.pyplot(fig)
            plt.close(fig)

        for col in ['family', 'city', 'transferred']:
            if col in df.columns:
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.countplot(x=col, data=df, ax=ax, order=df[col].value_counts().index)
                plt.xticks(rotation=45)
                plt.savefig(os.path.join(temp_dir, f"{dataset_type}_{col}_imbalance.png"))
                st.pyplot(fig)
                plt.close(fig)

        for col in ['family', 'city', 'state', 'store_nbr']:
            if col in df.columns and 'sales' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.boxplot(x=col, y='sales', data=df, ax=ax)
                plt.xticks(rotation=45)
                plt.savefig(os.path.join(temp_dir, f"{dataset_type}_sales_{col}.png"))
                st.pyplot(fig)
                plt.close(fig)

        if 'sales' in df.columns and 'onpromotion' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.boxplot(x='onpromotion', y='sales', data=df, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_promo.png"))
            st.pyplot(fig)
            plt.close(fig)

        if 'sales' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.histplot(df['sales'], bins=30, kde=True, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_dist.png"))
            st.pyplot(fig)
            plt.close(fig)

        if numeric_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_corr.png"))
            st.pyplot(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(3 * len(numeric_cols), 3))
            axes = [axes] if len(numeric_cols) == 1 else axes
            for i, col in enumerate(numeric_cols):
                sns.boxplot(x=df[col], ax=axes[i])
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_outliers.png"))
            st.pyplot(fig)
            plt.close(fig)

def preprocess_data(df, numeric_cols, categorical_cols, date_col, handle_outliers, scale, dataset_type):
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isna().sum() > 0:
            if col in numeric_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif col in categorical_cols:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            else:
                df_clean[col].fillna('Unknown', inplace=True)
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
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
    if date_col:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_clean['year'] = df_clean[date_col].dt.year
        df_clean['month'] = df_clean[date_col].dt.month
        df_clean['day'] = df_clean[date_col].dt.day
        df_clean['weekday'] = df_clean[date_col].dt.weekday
        df_clean['week'] = df_clean[date_col].dt.isocalendar().week
        df_clean['is_weekend'] = df_clean[date_col].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
        df_clean['season'] = pd.cut(df_clean['month'], bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4'])
        df_clean['sin_month'] = np.sin(2 * np.pi * df_clean['month'] / 12)
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype('category')
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col])
    if scale and numeric_cols:
        valid_cols = [col for col in numeric_cols if df_clean[col].std() > 1e-6]
        scaler = StandardScaler()
        if dataset_type == "train":
            df_clean[valid_cols] = scaler.fit_transform(df_clean[valid_cols])
            st.session_state['scaler'] = scaler
        elif dataset_type == "test" and 'scaler' in st.session_state:
            df_clean[valid_cols] = st.session_state['scaler'].transform(df_clean[valid_cols])
    df_clean = clean_text_columns(df_clean)
    return df_clean, initial_rows - df_clean.shape[0]

def engineer_features(train_df, test_df, numeric_cols, categorical_cols, target):
    train_fe, test_fe = train_df.copy(), test_df.copy()
    if 'onpromotion' in train_fe and target in train_fe:
        train_fe['sales_onpromo'] = train_fe[target] * train_fe['onpromotion']
        test_fe['sales_onpromo'] = test_fe['onpromotion'] * 0 if 'onpromotion' in test_fe else 0
    if 'onpromotion' in train_fe and 'is_weekend' in train_fe:
        train_fe['promo_weekend'] = train_fe['onpromotion'] * train_fe['is_weekend']
        test_fe['promo_weekend'] = test_fe['onpromotion'] * test_fe['is_weekend'] if 'onpromotion' in test_fe and 'is_weekend' in test_fe else 0
    for col in ['city', 'state']:
        if col in train_fe and target in train_fe:
            mean_map = train_fe.groupby(col)[target].mean().to_dict()
            train_fe[f'{col}_encoded'] = train_fe[col].map(mean_map)
            test_fe[f'{col}_encoded'] = test_fe[col].map(mean_map).fillna(train_fe[target].mean())
    for col in ['family', 'locale_name']:
        if col in train_fe:
            freq_map = train_fe[col].value_counts(normalize=True).to_dict()
            train_fe[f'{col}_encoded'] = train_fe[col].map(freq_map)
            test_fe[f'{col}_encoded'] = test_fe[col].map(freq_map).fillna(0)
    if 'store_nbr' in train_fe and 'month' in train_fe and target in train_fe:
        train_fe['avg_sales_store_month'] = train_fe.groupby(['store_nbr', 'month'])[target].transform('mean')
        store_month_means = train_fe.groupby(['store_nbr', 'month'])[target].mean().to_dict()
        test_fe['avg_sales_store_month'] = test_fe.apply(
            lambda x: store_month_means.get((x['store_nbr'], x['month']), train_fe[target].mean()), axis=1)
    for df in [train_fe, test_fe]:
        if 'description' in df:
            df['is_holiday'] = df['description'].str.contains('Holiday|Navidad', case=False, na=False).astype(int)
        for col in ['transferred', 'is_weekend', 'is_holiday']:
            if col in df:
                df[col] = df[col].astype(int)
        locale_order = {'National': 2, 'Regional': 1, 'Local': 0}
        if 'locale' in df:
            df['locale_encoded'] = df['locale'].map(locale_order).fillna(0)
        type_y_order = {'Holiday': 2, 'Event': 1, 'Bridge': 0}
        if 'type_y' in df:
            df['type_y_encoded'] = df['type_y'].map(type_y_order).fillna(0)
    if 'dcoilwtico' in train_fe:
        q25, q75 = train_fe['dcoilwtico'].quantile(0.25), train_fe['dcoilwtico'].quantile(0.75)
        bins = [-np.inf, q25, np.inf] if q25 == q75 else [-np.inf, q25, q75, np.inf]
        labels = ['low', 'high'] if q25 == q75 else ['low', 'medium', 'high']
        train_fe['dcoilwtico_bin'] = pd.cut(train_fe['dcoilwtico'], bins=bins, labels=labels)
        test_fe['dcoilwtico_bin'] = pd.cut(test_fe['dcoilwtico'], bins=bins, labels=labels) if 'dcoilwtico' in test_fe else None
    return train_fe, test_fe

def main():
    train_tab, test_tab = st.tabs(["Train", "Test"])
    
    with train_tab:
        train_file = st.file_uploader("Train Data", ['csv', 'parquet'], key="train")
        if train_file:
            train_df = load_data(train_file, "train")
            st.session_state['train_df'] = train_df
            st.dataframe(train_df.head(), height=100)
            
            with st.form("train_config"):
                date_col = st.selectbox("Date", ['None'] + list(train_df.columns), 
                                       index=train_df.columns.tolist().index('date') if 'date' in train_df.columns else 0)
                date_col = None if date_col == 'None' else date_col
                target_col = st.selectbox("Target", train_df.columns, 
                                         index=train_df.columns.tolist().index('sales') if 'sales' in train_df.columns else 0)
                numeric_cols, categorical_cols = detect_column_types(train_df, date_col)
                numeric_cols = st.multiselect("Numeric", train_df.columns, default=numeric_cols)
                categorical_cols = st.multiselect("Categorical", train_df.columns, default=categorical_cols)
                outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace'])
                outlier_method = outlier_method.lower() if outlier_method != 'None' else None
                scale = st.checkbox("Scale")
                st.form_submit_button("Apply", on_click=lambda: st.session_state.update({
                    'train_date_col': date_col, 'train_target_col': target_col, 'train_numeric_cols': numeric_cols,
                    'train_categorical_cols': categorical_cols, 'train_outlier_method': outlier_method, 'train_scale': scale
                }))
            
            with st.form("train_process"):
                if st.form_submit_button("Run"):
                    date_col = st.session_state.get('train_date_col')
                    target_col = st.session_state.get('train_target_col', 'sales')
                    numeric_cols = st.session_state.get('train_numeric_cols', detect_column_types(train_df, date_col)[0])
                    categorical_cols = st.session_state.get('train_categorical_cols', detect_column_types(train_df, date_col)[1])
                    outlier_method = st.session_state.get('train_outlier_method')
                    scale = st.session_state.get('train_scale', False)
                    explore_data(train_df, date_col, numeric_cols, categorical_cols, "train")
                    processed_df, duplicates_removed = preprocess_data(
                        train_df, numeric_cols, categorical_cols, date_col, outlier_method, scale, "train")
                    st.session_state['processed_train'] = processed_df
                    st.write(f"{duplicates_removed} duplicates removed, {processed_df.shape[0]} rows")
                    st.dataframe(processed_df.head(), height=100)
            
            if 'processed_train' in st.session_state:
                csv_data, mime = get_download_file(st.session_state['processed_train'], "train_processed.csv")
                st.download_button("Download Train", csv_data, "train_processed.csv", mime, key="train_download")

    with test_tab:
        test_file = st.file_uploader("Test Data", ['csv', 'parquet'], key="test")
        if test_file:
            test_df = load_data(test_file, "test")
            st.session_state['test_df'] = test_df
            st.dataframe(test_df.head(), height=100)
            
            with st.form("test_config"):
                date_col = st.selectbox("Date", ['None'] + list(test_df.columns), 
                                       index=test_df.columns.tolist().index('date') if 'date' in test_df.columns else 0)
                date_col = None if date_col == 'None' else date_col
                numeric_cols, categorical_cols = detect_column_types(test_df, date_col)
                numeric_cols = st.multiselect("Numeric", test_df.columns, default=numeric_cols)
                categorical_cols = st.multiselect("Categorical", test_df.columns, default=categorical_cols)
                outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace'])
                outlier_method = outlier_method.lower() if outlier_method != 'None' else None
                scale = st.checkbox("Scale")
                st.form_submit_button("Apply", on_click=lambda: st.session_state.update({
                    'test_date_col': date_col, 'test_numeric_cols': numeric_cols, 'test_categorical_cols': categorical_cols,
                    'test_outlier_method': outlier_method, 'test_scale': scale
                }))
            
            with st.form("test_process"):
                if st.form_submit_button("Run"):
                    date_col = st.session_state.get('test_date_col')
                    numeric_cols = st.session_state.get('test_numeric_cols', detect_column_types(test_df, date_col)[0])
                    categorical_cols = st.session_state.get('test_categorical_cols', detect_column_types(test_df, date_col)[1])
                    outlier_method = st.session_state.get('test_outlier_method')
                    scale = st.session_state.get('test_scale', False)
                    explore_data(test_df, date_col, numeric_cols, categorical_cols, "test")
                    processed_df, duplicates_removed = preprocess_data(
                        test_df, numeric_cols, categorical_cols, date_col, outlier_method, scale, "test")
                    st.session_state['processed_test'] = processed_df
                    st.write(f"{duplicates_removed} duplicates removed, {processed_df.shape[0]} rows")
                    st.dataframe(processed_df.head(), height=100)
            
            if 'processed_test' in st.session_state:
                csv_data, mime = get_download_file(st.session_state['processed_test'], "test_processed.csv")
                st.download_button("Download Test", csv_data, "test_processed.csv", mime, key="test_download")

    if 'processed_train' in st.session_state and 'processed_test' in st.session_state:
        with st.form("feature_engineering"):
            if st.form_submit_button("Run Features"):
                train_fe, test_fe = engineer_features(
                    st.session_state['processed_train'], st.session_state['processed_test'],
                    st.session_state.get('train_numeric_cols', []), st.session_state.get('train_categorical_cols', []),
                    st.session_state.get('train_target_col', 'sales'))
                st.session_state['train_fe'] = train_fe
                st.session_state['test_fe'] = test_fe
                st.dataframe(train_fe.head(), height=100)
                st.dataframe(test_fe.head(), height=100)
        
        if 'train_fe' in st.session_state:
            train_csv, train_mime = get_download_file(st.session_state['train_fe'], "train_fe.csv")
            st.download_button("Download Train Features", train_csv, "train_fe.csv", train_mime, key="train_fe_download")
        
        if 'test_fe' in st.session_state:
            test_csv, test_mime = get_download_file(st.session_state['test_fe'], "test_fe.csv")
            st.download_button("Download Test Features", test_csv, "test_fe.csv", test_mime, key="test_fe_download")

    st.markdown("**By Belal Khamis, Marwa Kotb, Mahmoud Sabry, Mohamed Samy, Hoda Magdy**")

if __name__ == "__main__":
    main()
