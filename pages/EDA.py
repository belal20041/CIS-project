import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import zscore
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

def load_data(train_file, test_file, date_col, target_col):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    train[date_col] = pd.to_datetime(train[date_col])
    test[date_col] = pd.to_datetime(test[date_col], format='%d-%m-%Y')
    train[['store_nbr', 'onpromotion']] = train[['store_nbr', 'onpromotion']].astype('int32')
    test[['store_nbr', 'onpromotion']] = test[['store_nbr', 'onpromotion']].astype('int32')
    train[target_col] = train[target_col].astype('float32')
    train.dropna(subset=[date_col], inplace=True)
    test.dropna(subset=[date_col], inplace=True)
    return train, test

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

def split_data(combined, date_col):
    train = combined[combined['is_train'] == 1]
    test = combined[combined['is_train'] == 0]
    train_set = train[train[date_col] <= TRAIN_END]
    val_set = train[(train[date_col] > TRAIN_END) & (train[date_col] <= VAL_END)]
    return train_set, val_set, test

def get_download_file(df, filename):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue(), 'text/csv'

def explore_data(train_set, date_col, target_col, numeric_cols, categorical_cols, dataset_type):
    temp_dir = tempfile.gettempdir()
    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape:", train_set.shape)
        st.write("Missing:", train_set.isna().sum().to_dict())
        st.write("Duplicates:", train_set.duplicated().sum())
    with col2:
        st.write("Types:", train_set.dtypes.to_dict())
        st.write("Uniques:", train_set.nunique().to_dict())

    with st.expander("Visualizations"):
        fig, ax = plt.subplots(figsize=(8, 3))
        msno.matrix(train_set, ax=ax)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_missing.png"))
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3))
        sns.lineplot(x=date_col, y=target_col, data=train_set, ax=ax)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_trends.png"))
        st.pyplot(fig)
        plt.close(fig)

        df_weekly = train_set.set_index(date_col)[target_col].resample('W').sum().reset_index()
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.lineplot(x=date_col, y=target_col, data=df_weekly, ax=ax)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_weekly.png"))
        st.pyplot(fig)
        plt.close(fig)

        train_set['month'] = train_set[date_col].dt.month
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(x='month', y=target_col, data=train_set, ax=ax)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_seasonal.png"))
        st.pyplot(fig)
        plt.close(fig)

        rolling = train_set.set_index(date_col)[target_col].rolling(window=30).agg(['mean', 'std']).dropna()
        fig, ax = plt.subplots(figsize=(8, 3))
        rolling['mean'].plot(ax=ax, label='Mean')
        rolling['std'].plot(ax=ax, label='Std', alpha=0.5)
        plt.legend()
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_rolling.png"))
        st.pyplot(fig)
        plt.close(fig)

        df_ts = train_set.set_index(date_col)[target_col].resample('M').sum()
        decomp = seasonal_decompose(df_ts, model='additive', period=12)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
        decomp.trend.plot(ax=ax1)
        decomp.seasonal.plot(ax=ax2)
        decomp.resid.plot(ax=ax3)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_decomp.png"))
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3))
        plot_acf(train_set[target_col].dropna(), lags=30, ax=ax)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_acf.png"))
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3))
        plot_pacf(train_set[target_col].dropna(), lags=30, ax=ax)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_pacf.png"))
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3))
        pd.plotting.lag_plot(train_set[target_col], lag=1, ax=ax)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_lag.png"))
        st.pyplot(fig)
        plt.close(fig)

        freq, psd = periodogram(train_set[target_col].dropna())
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(freq, psd)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_periodogram.png"))
        st.pyplot(fig)
        plt.close(fig)

        store, family = 1, 'AUTOMOTIVE'
        ts = train_set[(train_set['store_nbr'] == store) & (train_set['family'] == family)][target_col]
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
            if col in train_set.columns:
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.boxplot(x=col, y=target_col, data=train_set, ax=ax)
                plt.xticks(rotation=45)
                plt.savefig(os.path.join(temp_dir, f"{dataset_type}_sales_{col}.png"))
                st.pyplot(fig)
                plt.close(fig)

        if target_col in train_set.columns and 'onpromotion' in train_set.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.boxplot(x='onpromotion', y=target_col, data=train_set, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_promo.png"))
            st.pyplot(fig)
            plt.close(fig)

        if target_col in train_set.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.histplot(train_set[target_col], bins=30, kde=True, ax=ax)
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_dist.png"))
            st.pyplot(fig)
            plt.close(fig)

def main():
    train_file = st.file_uploader("Train Data", ['csv'], key="train")
    test_file = st.file_uploader("Test Data", ['csv'], key="test")
    if train_file and test_file:
        train = pd.read_csv(train_file)
        with st.form("config"):
            date_col = st.selectbox("Date", train.columns)
            target_col = st.selectbox("Target", train.columns)
            numeric_cols, categorical_cols = detect_column_types(train, date_col)
            numeric_cols = st.multiselect("Numeric", train.columns, default=numeric_cols)
            categorical_cols = st.multiselect("Categorical", train.columns, default=categorical_cols)
            outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace'])
            outlier_method = outlier_method.lower() if outlier_method != 'None' else None
            scale = st.checkbox("Scale")
            st.form_submit_button("Apply", on_click=lambda: st.session_state.update({
                'date_col': date_col, 'target_col': target_col, 'numeric_cols': numeric_cols,
                'categorical_cols': categorical_cols, 'outlier_method': outlier_method, 'scale': scale
            }))

        with st.form("process"):
            if st.form_submit_button("Run"):
                train, test = load_data(train_file, test_file, st.session_state['date_col'], st.session_state['target_col'])
                combined = prepare_data(train, test, st.session_state['date_col'], st.session_state['target_col'])
                combined = fill_missing(combined, st.session_state['target_col'])
                combined = add_features(combined, st.session_state['date_col'], st.session_state['target_col'])
                train_set, val_set, test = split_data(combined, st.session_state['date_col'])

                st.dataframe(train_set.head(), height=100)
                st.dataframe(val_set.head(), height=100)
                st.dataframe(test.head(), height=100)

                explore_data(train_set, st.session_state['date_col'], st.session_state['target_col'], 
                             st.session_state['numeric_cols'], st.session_state['categorical_cols'], "train")

                csv_data, mime = get_download_file(train_set, "train_processed.csv")
                st.download_button("Download Train", csv_data, "train_processed.csv", mime, key="train_download")
                csv_data, mime = get_download_file(val_set, "val_processed.csv")
                st.download_button("Download Validation", csv_data, "val_processed.csv", mime, key="val_download")
                csv_data, mime = get_download_file(test, "test_processed.csv")
                st.download_button("Download Test", csv_data, "test_processed.csv", mime, key="test_download")

    st.markdown("By Belal Khamis@all rights reserved")

if __name__ == "__main__":
    main()
