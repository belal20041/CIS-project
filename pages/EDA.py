import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy.signal import periodogram
from io import BytesIO
import os
import tempfile

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Time Series Analysis</h1>", unsafe_allow_html=True)

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
        file.seek(0)  # Rewind file buffer
    df = pd.read_csv(file)
    df[date_col] = pd.to_datetime(df[date_col])
    df[['store_nbr', 'onpromotion']] = df[['store_nbr', 'onpromotion']].astype('int32')
    if target_col and target_col in df.columns:
        df[target_col] = df[target_col].astype('float32')
    df.dropna(subset=[date_col], inplace=True)
    return df

def prepare_data(train, test, date_col, target_col):
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test]).sort_values(['store_nbr', 'family', date_col])
    agg_dict = {target_col: 'mean', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first'}
    combined = combined.groupby(['store_nbr', 'family', date_col]).agg(agg_dict).reset_index()
    combined = combined.astype({'store_nbr': 'int32', 'family': 'category', date_col: 'datetime64[ns]', target_col: 'float32', 'onpromotion': 'int32', 'is_train': 'int8'})
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

    with st.expander("Visualizations"):
        fig, ax = plt.subplots(figsize=(8, 3))
        msno.matrix(df, ax=ax)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_missing.png"))
        st.pyplot(fig)
        plt.close(fig)

        if target_col and target_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.lineplot(x=date_col, y=target_col, data=df, ax=ax)
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_trends.png"))
            st.pyplot(fig)
            plt.close(fig)

            df_weekly = df.set_index(date_col)[target_col].resample('W').sum().reset_index()
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.lineplot(x=date_col, y=target_col, data=df_weekly, ax=ax)
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_weekly.png"))
            st.pyplot(fig)
            plt.close(fig)

            df['month'] = df[date_col].dt.month
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.boxplot(x='month', y=target_col, data=df, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_seasonal.png"))
            st.pyplot(fig)
            plt.close(fig)

            rolling = df.set_index(date_col)[target_col].rolling(window=30).agg(['mean', 'std']).dropna()
            fig, ax = plt.subplots(figsize=(8, 3))
            rolling['mean'].plot(ax=ax, label='Mean')
            rolling['std'].plot(ax=ax, label='Std', alpha=0.5)
            plt.legend()
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_rolling.png"))
            st.pyplot(fig)
            plt.close(fig)

            df_ts = df.set_index(date_col)[target_col].resample('M').sum()
            decomp = seasonal_decompose(df_ts, model='additive', period=12)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
            decomp.trend.plot(ax=ax1)
            decomp.seasonal.plot(ax=ax2)
            decomp.resid.plot(ax=ax3)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_decomp.png"))
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 3))
            plot_acf(df[target_col].dropna(), lags=30, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_acf.png"))
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 3))
            plot_pacf(df[target_col].dropna(), lags=30, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_pacf.png"))
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 3))
            pd.plotting.lag_plot(df[target_col], lag=1, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_lag.png"))
            st.pyplot(fig)
            plt.close(fig)

            freq, psd = periodogram(df[target_col].dropna())
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(freq, psd)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_periodogram.png"))
            st.pyplot(fig)
            plt.close(fig)

            store, family = 1, 'AUTOMOTIVE'
            ts = df[(df['store_nbr'] == store) & (df['family'] == family)][target_col]
            fig, ax = plt.subplots(figsize=(8, 3))
            plt.plot(ts, label='Sales')
            plt.plot(ts.rolling(30).mean(), label='30-Day Mean', color='red')
            plt.plot(ts.rolling(30).std(), label='30-Day Std', color='black')
            plt.legend()
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_stationarity.png"))
            st.pyplot(fig)
            plt.close(fig)

            ts_diff = ts.diff().dropna()
            fig, ax = plt.subplots(figsize=(8, 3))
            plot_acf(ts_diff, lags=20, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_acf_diff.png"))
            st.pyplot(fig)
            plt.close(fig)

        for col in ['family', 'store_nbr']:
            if col in df.columns and target_col and target_col in df.columns:
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.boxplot(x=col, y=target_col, data=df, ax=ax)
                plt.xticks(rotation=45)
                plt.savefig(os.path.join(temp_dir, f"{dataset_type}_sales_{col}.png"))
                st.pyplot(fig)
                plt.close(fig)

        if target_col and target_col in df.columns and 'onpromotion' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.boxplot(x='onpromotion', y=target_col, data=df, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_promo.png"))
            st.pyplot(fig)
            plt.close(fig)

        if target_col and target_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.histplot(df[target_col], bins=30, kde=True, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_dist.png"))
            st.pyplot(fig)
            plt.close(fig)

def main():
    train_tab, test_tab = st.tabs(["Train", "Test"])

    with train_tab:
        train_file = st.file_uploader("Train Data", ['csv'], key="train")
        if train_file:
            if 'train_file' in st.session_state and st.session_state['train_file'] != train_file:
                for key in ['train_date_col', 'train_target_col', 'train_numeric_cols', 'train_categorical_cols', 'train_outlier_method', 'train_scale', 'train_configured', 'train_df']:
                    st.session_state.pop(key, None)
            with st.form("train_config"):
                train = load_data(train_file, 'date', None)
                st.dataframe(train.head(), height=100)
                date_col = st.selectbox("Date", train.columns, index=train.columns.tolist().index('date') if 'date' in train.columns else 0, key="train_date")
                target_col = st.selectbox("Target", train.columns, key="train_target")
                numeric_cols, categorical_cols = detect_column_types(train, date_col)
                numeric_cols = st.multiselect("Numeric", train.columns, default=['id', 'store_nbr', 'onpromotion', 'cluster', 'dcoilwtico'] if 'store_nbr' in train.columns else numeric_cols, key="train_numeric")
                categorical_cols = st.multiselect("Categorical", train.columns, default=['store_nbr', 'family', 'onpromotion', 'city', 'state', 'type_x', 'cluster', 'type_y', 'locale', 'locale_name', 'description', 'transferred', 'dcoilwtico'] if 'family' in train.columns else categorical_cols, key="train_categorical")
                outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace'], index=2, key="train_outlier")
                outlier_method = outlier_method.lower() if outlier_method != 'None' else None
                scale = st.checkbox("Scale", key="train_scale")
                st.form_submit_button("Apply", on_click=lambda: st.session_state.update({
                    'train_date_col': date_col, 'train_target_col': target_col, 'train_numeric_cols': numeric_cols,
                    'train_categorical_cols': categorical_cols, 'train_outlier_method': outlier_method, 'train_scale': scale,
                    'train_configured': True, 'train_file': train_file
                }))

            if 'train_configured' in st.session_state and st.session_state['train_configured'] and 'train_file' in st.session_state:
                with st.form("train_process"):
                    if st.form_submit_button("Run"):
                        train = load_data(st.session_state['train_file'], st.session_state['train_date_col'], st.session_state['train_target_col'])
                        st.session_state['train_df'] = train
                        explore_data(train, st.session_state['train_date_col'], st.session_state['train_target_col'], 
                                     st.session_state['train_numeric_cols'], st.session_state['train_categorical_cols'], "train")
                        st.dataframe(train.head(), height=100)
                csv_data, mime = get_download_file(train, "train_processed.csv")
                st.download_button("Download Train", csv_data, "train_processed.csv", mime, key="train_download")

    with test_tab:
        test_file = st.file_uploader("Test Data", ['csv'], key="test")
        if test_file:
            if 'test_file' in st.session_state and st.session_state['test_file'] != test_file:
                for key in ['test_date_col', 'test_numeric_cols', 'test_categorical_cols', 'test_outlier_method', 'test_scale', 'test_configured', 'test_df']:
                    st.session_state.pop(key, None)
            with st.form("test_config"):
                test = load_data(test_file, 'date', None)
                st.dataframe(test.head(), height=100)
                date_col = st.selectbox("Date", test.columns, index=test.columns.tolist().index('date') if 'date' in test.columns else 0, key="test_date")
                numeric_cols, categorical_cols = detect_column_types(test, date_col)
                numeric_cols = st.multiselect("Numeric", test.columns, default=['id', 'store_nbr', 'onpromotion', 'cluster', 'dcoilwtico'] if 'store_nbr' in test.columns else numeric_cols, key="test_numeric")
                categorical_cols = st.multiselect("Categorical", test.columns, default=['store_nbr', 'family', 'onpromotion', 'city', 'state', 'type_x', 'cluster', 'type_y', 'locale', 'locale_name', 'description', 'transferred', 'dcoilwtico'] if 'family' in test.columns else categorical_cols, key="test_categorical")
                outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace'], index=2, key="test_outlier")
                outlier_method = outlier_method.lower() if outlier_method != 'None' else None
                scale = st.checkbox("Scale", key="test_scale")
                st.form_submit_button("Apply", on_click=lambda: st.session_state.update({
                    'test_date_col': date_col, 'test_numeric_cols': numeric_cols, 'test_categorical_cols': categorical_cols,
                    'test_outlier_method': outlier_method, 'test_scale': scale, 'test_configured': True, 'test_file': test_file
                }))

            if 'test_configured' in st.session_state and st.session_state['test_configured'] and 'test_file' in st.session_state:
                with st.form("test_process"):
                    if st.form_submit_button("Run"):
                        test = load_data(st.session_state['test_file'], st.session_state['test_date_col'], None)
                        st.session_state['test_df'] = test
                        explore_data(test, st.session_state['test_date_col'], st.session_state.get('train_target_col', None), 
                                     st.session_state['test_numeric_cols'], st.session_state['test_categorical_cols'], "test")
                        st.dataframe(test.head(), height=100)
                csv_data, mime = get_download_file(test, "test_processed.csv")
                st.download_button("Download Test", csv_data, "test_processed.csv", mime, key="test_download")

    if 'train_df' in st.session_state and 'test_df' in st.session_state:
        with st.form("feature_engineering"):
            if st.form_submit_button("Run Features"):
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
                csv_data, mime = get_download_file(train_set, "train_fe.csv")
                st.download_button("Download Train Features", csv_data, "train_fe.csv", mime, key="train_fe_download")
                csv_data, mime = get_download_file(val_set, "val_fe.csv")
                st.download_button("Download Validation Features", csv_data, "val_fe.csv", mime, key="val_fe_download")
                csv_data, mime = get_download_file(test, "test_fe.csv")
                st.download_button("Download Test Features", csv_data, "test_fe.csv", mime, key="test_fe_download")

    st.markdown("**By Belal Khamis, Marwa Kotb, Mahmoud Sabry, Mohamed Samy, Hoda Magdy**")

if __name__ == "__main__":
    main()
