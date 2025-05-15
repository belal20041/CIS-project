import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
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
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')  # Removed strict format to handle varying date formats
    df[['store_nbr', 'onpromotion']] = df[['store_nbr', 'onpromotion']].astype('int32')
    if target_col and target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').astype('float32')
    df.dropna(subset=[date_col], inplace=True)
    return df

def prepare_data(train, test, date_col, target_col):
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test]).sort_values(['store_nbr', 'family', date_col])
    agg_dict = {target_col: 'sum', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first'}
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

def reclassify_family(df):
    family_map = {
        'AUTOMOTIVE': 'Tools', 'HARDWARE': 'Tools', 'LAWN AND GARDEN': 'Tools', 'PLAYERS AND ELECTRONICS': 'Tools',
        'BEAUTY': 'LifeStyle', 'LINGERIE': 'LifeStyle', 'LADIESWEAR': 'LifeStyle', 'PERSONAL CARE': 'LifeStyle',
        'CELEBRATION': 'LifeStyle', 'MAGAZINES': 'LifeStyle', 'BOOKS': 'LifeStyle', 'BABY CARE': 'LifeStyle',
        'HOME APPLIANCES': 'Home', 'HOME AND KITCHEN I': 'Home', 'HOME AND KITCHEN II': 'Home',
        'HOME CARE': 'Home', 'SCHOOL AND OFFICE SUPPLIES': 'Home',
        'GROCERY II': 'Food', 'PET SUPPLIES': 'Food', 'SEAFOOD': 'Food', 'LIQUOR,WINE,BEER': 'Food',
        'DELI': 'Daily', 'EGGS': 'Daily'
    }
    df['family'] = df['family'].replace(family_map)
    return df

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

        # 3. Weekly Sales Trends
        df_weekly = df.set_index(date_col)['sales'].resample('W').sum().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_weekly[date_col], df_weekly['sales'], color='blue')
        ax.grid(True, alpha=0.3)
        ax.set_title("Weekly Sales Trends")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_weekly.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 4. Sales by Reclassified Product Family
        df = reclassify_family(df.copy())
        sales_by_family = df.groupby('family')['sales'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(25, 15))
        sns.barplot(x=sales_by_family.values, y=sales_by_family.index, ax=ax, color='skyblue')
        ax.grid(True, alpha=0.3)
        ax.set_title("Average Sales by Product Category")
        ax.set_xlabel("Average Sales")
        ax.set_ylabel("Product Family")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_family_sales.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 5. Sales by Store Number
        sales_by_store = df.groupby('store_nbr')['sales'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(25, 15))
        sns.barplot(x=sales_by_store.index, y=sales_by_store.values, ax=ax, color='skyblue', order=sales_by_store.index)
        ax.grid(True, alpha=0.3)
        ax.set_title("Average Sales by Store Number")
        ax.set_xlabel("Store Number")
        ax.set_ylabel("Average Sales")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_store_sales.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 6. Sales by City-State
        df['city_state'] = df['city'] + '_' + df['state']
        sales_by_city_state = df.groupby('city_state')['sales'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(30, 20))
        sns.barplot(x=sales_by_city_state.values, y=sales_by_city_state.index, ax=ax, color='skyblue')
        ax.grid(True, alpha=0.3)
        ax.set_title("Average Sales by City-State")
        ax.set_xlabel("Average Sales")
        ax.set_ylabel("City-State")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_city_state_sales.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 7. Sales by Type-Locale
        df['type_locale'] = df['type_y'] + '_' + df['locale']
        sales_by_type_locale = df.groupby('type_locale')['sales'].mean()
        fig, ax = plt.subplots(figsize=(20, 10))
        sales_by_type_locale.plot.pie(autopct='%1.1f%%', ax=ax, startangle=90, colors=sns.color_palette('muted'))
        ax.set_title("Sales Distribution by Type-Locale")
        ax.set_ylabel("")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_type_locale_pie.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 8. Impact of Promotions on Sales
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

        # 9. Sales vs. Oil Price
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

        # 10. Monthly Sales Seasonality
        df['month'] = df[date_col].dt.month
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

        # 11. Day-of-Week Sales Patterns
        df['dow'] = df[date_col].dt.dayofweek
        sales_by_dow = df.groupby(['dow', 'family'])['sales'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=sales_by_dow, x='dow', y='sales', hue='family', ax=ax, palette='muted')
        ax.grid(True, alpha=0.3)
        ax.set_title("Average Sales by Day of Week (by Family)")
        ax.set_xlabel("Day of Week (0=Mon, 6=Sun)")
        ax.set_ylabel("Average Sales")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_dow_sales.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 12. Seasonal Decomposition (Monthly)
        df_ts = df.set_index(date_col)['sales'].resample('M').sum()
        if len(df_ts) >= 24:  # Ensure enough data for decomposition (at least 2 years)
            decomp = seasonal_decompose(df_ts, model='additive', period=12)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
            decomp.trend.plot(ax=ax1, color='blue')
            ax1.set_title("Trend Component")
            ax1.grid(True, alpha=0.3)
            decomp.seasonal.plot(ax=ax2, color='blue')
            ax2.set_title("Seasonal Component")
            ax2.grid(True, alpha=0.3)
            decomp.resid.plot(ax=ax3, color='blue')
            ax3.set_title("Residual Component")
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(temp_dir, f"{dataset_type}_decomp.png"))
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("Not enough data for seasonal decomposition (requires at least 24 months).")

        # 13. Autocorrelation (ACF)
        n_lags = 30
        acf_vals, acf_confint = acf(df[target_col].dropna(), nlags=n_lags, alpha=0.05, fft=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.stem(range(len(acf_vals)), acf_vals)
        plt.fill_between(range(len(acf_vals)), acf_confint[:, 0] - acf_vals, acf_confint[:, 1] - acf_vals, alpha=0.2)
        plt.axhline(0, color='black', linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.title("Autocorrelation (ACF) of Sales")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_acf.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 14. Partial Autocorrelation (PACF)
        pacf_vals, pacf_confint = pacf(df[target_col].dropna(), nlags=n_lags, alpha=0.05)
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.stem(range(len(pacf_vals)), pacf_vals)
        plt.fill_between(range(len(pacf_vals)), pacf_confint[:, 0] - pacf_vals, pacf_confint[:, 1] - pacf_vals, alpha=0.2)
        plt.axhline(0, color='black', linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.title("Partial Autocorrelation (PACF) of Sales")
        plt.xlabel("Lag")
        plt.ylabel("Partial Autocorrelation")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_pacf.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 15. Lag Plot
        lag = 1
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.scatter(df[target_col].shift(lag), df[target_col], alpha=0.5, color='blue')
        plt.grid(True, alpha=0.3)
        plt.title(f"Lag Plot (Lag={lag})")
        plt.xlabel(f"Sales (t-{lag})")
        plt.ylabel("Sales (t)")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_lag.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 16. Periodogram
        freq, psd = periodogram(df[target_col].dropna())
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.plot(freq, psd, color='blue')
        plt.axhline(0, color='black', linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.title("Periodogram of Sales")
        plt.xlabel("Frequency")
        plt.ylabel("Power Spectral Density")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_periodogram.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 17. Rolling Statistics for Stationarity
        rolling = df.set_index(date_col)['sales'].rolling(window=30).agg(['mean', 'std']).dropna()
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.plot(rolling.index, rolling['mean'], label='Mean', color='blue')
        plt.plot(rolling.index, rolling['std'], label='Std', color='orange', alpha=0.5)
        plt.axhline(0, color='black', linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.title("Rolling Mean and Std (30 Days)")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.savefig(os.path.join(temp_dir, f"{dataset_type}_rolling.png"))
        st.pyplot(fig)
        plt.close(fig)

        # 18. Heatmap: Sales Across Stores and Families
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
                train = load_data(train_file, 'date', 'sales')
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
                categorical_cols = st.multiselect("Categorical Columns", test.columns, default=['family', 'city', 'state', 'type_x', 'type_y', 'locale', 'locale_name', 'description', 'transferred'] if all(col in train.columns for col in ['family', 'city']) else categorical_cols, key="test_categorical")
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
