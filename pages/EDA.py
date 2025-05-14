import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from io import BytesIO

st.set_page_config(page_title="Milestone 1: Data Analysis", layout="wide")
st.markdown("<h1 style='text-align: center;'>Data Collection & Preprocessing</h1>", unsafe_allow_html=True)

def load_data(file, dataset_type="train"):
    """Load dataset from uploaded file (CSV or Parquet)."""
    if file:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.parquet'):
                df = pd.read_parquet(file)
            else:
                st.error("Unsupported format. Use CSV or Parquet.")
                return None
            df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]
            df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
            return df
        except Exception as e:
            st.error(f"Failed to load {dataset_type} data: {e}")
            return None
    return None

def detect_column_types(df, date_col=None):
    """Detect numeric and categorical columns."""
    numeric_cols = df.select_dtypes(['int64', 'float64']).columns.tolist()
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
    """Perform EDA with key visualizations, including time series and imbalance."""
    st.markdown(f"### {dataset_type.capitalize()} Data Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape**:", df.shape)
        st.write("**Missing Values**:", df.isna().sum().to_dict())
        st.write("**Duplicates**:", df.duplicated().sum())
    with col2:
        st.write("**Data Types**:", df.dtypes.to_dict())
        st.write("**Unique Values**:", df.nunique().to_dict())

    # Missing values visualization
    st.markdown("**Missing Values Matrix**")
    fig, ax = plt.subplots(figsize=(8, 3))
    msno.matrix(df, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    # Time series visualizations
    if date_col and 'sales' in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col)

        # Sales trends
        st.markdown("**Sales Trends**")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.lineplot(x=date_col, y='sales', data=df, ax=ax)
        ax.set_title("Sales Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

        # Monthly seasonality
        df['month'] = df[date_col].dt.month
        st.markdown("**Monthly Seasonality**")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(x='month', y='sales', data=df, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        # Rolling mean and std
        st.markdown("**Rolling Mean and Std (30-day window)**")
        rolling = df.set_index(date_col)['sales'].rolling(window=30).agg(['mean', 'std']).dropna()
        fig, ax = plt.subplots(figsize=(8, 3))
        rolling['mean'].plot(ax=ax, label='Rolling Mean')
        rolling['std'].plot(ax=ax, label='Rolling Std', alpha=0.5)
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

        # Seasonal decomposition
        st.markdown("**Seasonal Decomposition (Yearly)**")
        try:
            df_ts = df.set_index(date_col)['sales'].resample('M').sum()
            decomp = seasonal_decompose(df_ts, model='additive', period=12)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
            decomp.trend.plot(ax=ax1, title='Trend')
            decomp.seasonal.plot(ax=ax2, title='Seasonal')
            decomp.resid.plot(ax=ax3, title='Residual')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Seasonal decomposition failed: {e}")

        # Autocorrelation
        st.markdown("**Autocorrelation Plot**")
        fig, ax = plt.subplots(figsize=(8, 3))
        plot_acf(df['sales'].dropna(), lags=30, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # Sales distribution by categories
    for col in ['family', 'city', 'state', 'store_nbr']:
        if col in df.columns and 'sales' in df.columns:
            st.markdown(f"**Sales by {col.capitalize()}**")
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.boxplot(data=df, x=col, y='sales', ax=ax)
            ax.set_title(f"Sales by {col.capitalize()}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

    # Sales vs promotions
    if 'sales' in df.columns and 'onpromotion' in df.columns:
        st.markdown("**Sales vs Promotions**")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(x='onpromotion', y='sales', data=df, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # Sales distribution
    if 'sales' in df.columns:
        st.markdown("**Sales Distribution**")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(df['sales'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # Correlation heatmap
    if numeric_cols:
        st.markdown("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # Outliers
    if numeric_cols:
        st.markdown("**Outliers**")
        fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 4))
        axes = [axes] if len(numeric_cols) == 1 else axes
        for i, col in enumerate(numeric_cols):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers_iqr = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
            z_scores = zscore(df[col].dropna())
            outliers_zscore = df.iloc[df[col].dropna().index][abs(z_scores) > 3][col]
            st.write(f"{col}: IQR Outliers = {len(outliers_iqr)}, Z-score Outliers = {len(outliers_zscore)}")

    # Data imbalance for categorical columns
    st.markdown("**Categorical Data Imbalance**")
    for col in ['family', 'city', 'transferred'] if categorical_cols else []:
        if col in df.columns:
            st.markdown(f"**Distribution of {col.capitalize()}**")
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.countplot(x=col, data=df, ax=ax, order=df[col].value_counts().index)
            ax.set_title(f"Distribution of {col.capitalize()}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

def preprocess_data(df, numeric_cols, categorical_cols, date_col=None, handle_outliers=None, scale=False, dataset_type="train"):
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
        df_clean['sin_month'] = np.sin(2 * np.pi * df_clean['month'] / 12)

    # Convert categorical columns to category dtype
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')

    # Ensure numeric columns are numeric
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Scale numeric columns
    if scale and numeric_cols:
        valid_cols = [col for col in numeric_cols if col in df_clean and df_clean[col].std() > 1e-6]
        if valid_cols:
            scaler = StandardScaler()
            if dataset_type == "train":
                df_clean[valid_cols] = scaler.fit_transform(df_clean[valid_cols])
                st.session_state['scaler'] = scaler
            elif dataset_type == "test" and 'scaler' in st.session_state:
                df_clean[valid_cols] = st.session_state['scaler'].transform(df_clean[valid_cols])

    # Clean text columns
    text_cols = ['family', 'city', 'state', 'cluster', 'type_y', 'locale', 'locale_name', 'description', 'transferred']
    df_clean = clean_text_columns(df_clean, text_cols)

    return df_clean, initial_rows - df_clean.shape[0]

def engineer_features(train_df, test_df, numeric_cols, categorical_cols, target='sales'):
    """Engineer features for train and test datasets."""
    train_fe, test_fe = train_df.copy(), test_df.copy()

    # Interaction terms
    if 'onpromotion' in train_fe and target in train_fe:
        train_fe['sales_onpromo'] = train_fe[target] * train_fe['onpromotion']
        test_fe['sales_onpromo'] = test_fe['onpromotion'] * 0 if 'onpromotion' in test_fe else 0
    if 'onpromotion' in train_fe and 'is_weekend' in train_fe:
        train_fe['promo_weekend'] = train_fe['onpromotion'] * train_fe['is_weekend']
        test_fe['promo_weekend'] = test_fe['onpromotion'] * test_fe['is_weekend'] if 'onpromotion' in test_fe and 'is_weekend' in test_fe else 0

    # Mean encoding
    for col in ['city', 'state']:
        if col in train_fe and target in train_fe:
            mean_map = train_fe.groupby(col)[target].mean().to_dict()
            train_fe[f'{col}_encoded'] = train_fe[col].map(mean_map)
            test_fe[f'{col}_encoded'] = test_fe[col].map(mean_map).fillna(train_fe[target].mean())

    # Frequency encoding
    for col in ['family', 'locale_name']:
        if col in train_fe:
            freq_map = train_fe[col].value_counts(normalize=True).to_dict()
            train_fe[f'{col}_encoded'] = train_fe[col].map(freq_map)
            test_fe[f'{col}_encoded'] = test_fe[col].map(freq_map).fillna(0)

    # Average sales per store and month
    if 'store_nbr' in train_fe and 'month' in train_fe and target in train_fe:
        train_fe['avg_sales_store_month'] = train_fe.groupby(['store_nbr', 'month'])[target].transform('mean')
        store_month_means = train_fe.groupby(['store_nbr', 'month'])[target].mean().to_dict()
        test_fe['avg_sales_store_month'] = test_fe.apply(
            lambda x: store_month_means.get((x['store_nbr'], x['month']), train_fe[target].mean()), axis=1)

    # Holiday and encoding
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

    # Binning
    if 'dcoilwtico' in train_fe:
        q25, q75 = train_fe['dcoilwtico'].quantile(0.25), train_fe['dcoilwtico'].quantile(0.75)
        bins = [-np.inf, q25, np.inf] if q25 == q75 else [-np.inf, q25, q75, np.inf]
        labels = ['low', 'high'] if q25 == q75 else ['low', 'medium', 'high']
        train_fe['dcoilwtico_bin'] = pd.cut(train_fe['dcoilwtico'], bins=bins, labels=labels)
        test_fe['dcoilwtico_bin'] = pd.cut(test_fe['dcoilwtico'], bins=bins, labels=labels) if 'dcoilwtico' in test_fe else None

    return train_fe, test_fe

def get_download_file(df, filename):
    """Generate downloadable CSV file."""
    try:
        if df is None or df.empty:
            raise ValueError("Dataframe is empty or None")
        buf = BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf.getvalue(), 'text/csv'
    except Exception as e:
        st.error(f"Failed to generate download file: {e}")
        return None, None

def main():
    """Main Streamlit app."""
    with st.expander("ℹ️ Project Info"):
        st.markdown("""
        **Objective**: Collect, explore, and preprocess historical sales data for modeling.
        **Tasks**:
        - Collect: Historical sales with features (sales, date, promotions, holidays).
        - Explore: Trends, seasonality, missing values, outliers, data imbalance.
        - Preprocess: Handle missing values, duplicates, add time features, scale data.
        **Team**:
        - Belal Khamis: Notebook Outlines, Missing Values
        - Marwa Kotb: Duplicates
        - Mahmoud Sabry: Outliers
        - Mohamed Samy: Standardization, Time Features
        - Hoda Magdy: Text Cleaning
        """)

    # File uploaders
    train_tab, test_tab = st.tabs(["Train Data", "Test Data"])

    with train_tab:
        train_file = st.file_uploader("Upload Train Data", type=['csv', 'parquet'], key="train")
        if train_file:
            train_df = load_data(train_file, "train")
            if train_df is not None:
                st.session_state['train_df'] = train_df
                st.dataframe(train_df.head(), height=150)

                with st.form("train_config"):
                    date_col = st.selectbox("Date Column", ['None'] + list(train_df.columns), 
                                           index=train_df.columns.tolist().index('date') if 'date' in train_df.columns else 0)
                    date_col = None if date_col == 'None' else date_col
                    target_col = st.selectbox("Target", train_df.columns, 
                                             index=train_df.columns.tolist().index('sales') if 'sales' in train_df.columns else 0)
                    numeric_cols, categorical_cols = detect_column_types(train_df, date_col)
                    numeric_cols = st.multiselect("Numeric Columns", train_df.columns, default=numeric_cols)
                    categorical_cols = st.multiselect("Categorical Columns", train_df.columns, default=categorical_cols)
                    outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace'], key="train_outliers")
                    outlier_method = outlier_method.lower() if outlier_method != 'None' else None
                    scale = st.checkbox("Scale Numerics")
                    st.form_submit_button("Apply", on_click=lambda: st.session_state.update({
                        'train_date_col': date_col,
                        'train_target_col': target_col,
                        'train_numeric_cols': numeric_cols,
                        'train_categorical_cols': categorical_cols,
                        'train_outlier_method': outlier_method,
                        'train_scale': scale
                    }))

                with st.form("train_process"):
                    st.markdown("**Process Train Data**")
                    if st.form_submit_button("Run"):
                        date_col = st.session_state.get('train_date_col', None)
                        target_col = st.session_state.get('train_target_col', 'sales')
                        numeric_cols = st.session_state.get('train_numeric_cols', detect_column_types(train_df, date_col)[0])
                        categorical_cols = st.session_state.get('train_categorical_cols', detect_column_types(train_df, date_col)[1])
                        outlier_method = st.session_state.get('train_outlier_method', None)
                        scale = st.session_state.get('train_scale', False)

                        # Exploration
                        explore_data(train_df, date_col, numeric_cols, categorical_cols, "train")

                        # Preprocessing
                        processed_df, duplicates_removed = preprocess_data(
                            train_df, numeric_cols, categorical_cols, date_col, outlier_method, scale, "train")
                        st.session_state['processed_train'] = processed_df
                        st.write(f"Processed: {duplicates_removed} duplicates removed, {processed_df.shape[0]} rows remain")
                        st.dataframe(processed_df.head(), height=150)

                # Download processed train data (outside form)
                if 'processed_train' in st.session_state:
                    csv_data, mime = get_download_file(st.session_state['processed_train'], "train_processed.csv")
                    if csv_data and mime:
                        try:
                            st.download_button(
                                label="Download Processed Train Data",
                                data=csv_data,
                                file_name="train_processed.csv",
                                mime=mime,
                                key="train_download"
                            )
                        except Exception as e:
                            st.error(f"Download button failed: {e}")

    with test_tab:
        test_file = st.file_uploader("Upload Test Data", type=['csv', 'parquet'], key="test")
        if test_file:
            test_df = load_data(test_file, "test")
            if test_df is not None:
                st.session_state['test_df'] = test_df
                st.dataframe(test_df.head(), height=150)

                with st.form("test_config"):
                    date_col = st.selectbox("Date Column", ['None'] + list(test_df.columns), 
                                           index=test_df.columns.tolist().index('date') if 'date' in test_df.columns else 0)
                    date_col = None if date_col == 'None' else date_col
                    numeric_cols, categorical_cols = detect_column_types(test_df, date_col)
                    numeric_cols = st.multiselect("Numeric Columns", test_df.columns, default=numeric_cols)
                    categorical_cols = st.multiselect("Categorical Columns", test_df.columns, default=categorical_cols)
                    outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace'], key="test_outliers")
                    outlier_method = outlier_method.lower() if outlier_method != 'None' else None
                    scale = st.checkbox("Scale Numerics")
                    st.form_submit_button("Apply", on_click=lambda: st.session_state.update({
                        'test_date_col': date_col,
                        'test_numeric_cols': numeric_cols,
                        'test_categorical_cols': categorical_cols,
                        'test_outlier_method': outlier_method,
                        'test_scale': scale
                    }))

                with st.form("test_process"):
                    st.markdown("**Process Test Data**")
                    if st.form_submit_button("Run"):
                        date_col = st.session_state.get('test_date_col', None)
                        numeric_cols = st.session_state.get('test_numeric_cols', detect_column_types(test_df, date_col)[0])
                        categorical_cols = st.session_state.get('test_categorical_cols', detect_column_types(test_df, date_col)[1])
                        outlier_method = st.session_state.get('test_outlier_method', None)
                        scale = st.session_state.get('test_scale', False)

                        # Exploration
                        explore_data(test_df, date_col, numeric_cols, categorical_cols, "test")

                        # Preprocessing
                        processed_df, duplicates_removed = preprocess_data(
                            test_df, numeric_cols, categorical_cols, date_col, outlier_method, scale, "test")
                        st.session_state['processed_test'] = processed_df
                        st.write(f"Processed: {duplicates_removed} duplicates removed, {processed_df.shape[0]} rows remain")
                        st.dataframe(processed_df.head(), height=150)

                # Download processed test data (outside form)
                if 'processed_test' in st.session_state:
                    csv_data, mime = get_download_file(st.session_state['processed_test'], "test_processed.csv")
                    if csv_data and mime:
                        try:
                            st.download_button(
                                label="Download Processed Test Data",
                                data=csv_data,
                                file_name="test_processed.csv",
                                mime=mime,
                                key="test_download"
                            )
                        except Exception as e:
                            st.error(f"Download button failed: {e}")

    # Feature Engineering
    if 'processed_train' in st.session_state and 'processed_test' in st.session_state:
        st.markdown("**Feature Engineering**")
        with st.form("feature_engineering"):
            if st.form_submit_button("Run Feature Engineering"):
                train_fe, test_fe = engineer_features(
                    st.session_state['processed_train'],
                    st.session_state['processed_test'],
                    st.session_state.get('train_numeric_cols', []),
                    st.session_state.get('train_categorical_cols', []),
                    st.session_state.get('train_target_col', 'sales')
                )
                st.session_state['train_fe'] = train_fe
                st.session_state['test_fe'] = test_fe

                st.markdown("**Feature Engineered Train Data**")
                st.dataframe(train_fe.head(), height=150)

                st.markdown("**Feature Engineered Test Data**")
                st.dataframe(test_fe.head(), height=150)

        # Download feature-engineered data (outside form)
        if 'train_fe' in st.session_state:
            train_csv, train_mime = get_download_file(st.session_state['train_fe'], "train_fe.csv")
            if train_csv and train_mime:
                try:
                    st.download_button(
                        label="Download Feature Engineered Train Data",
                        data=train_csv,
                        file_name="train_fe.csv",
                        mime=train_mime,
                        key="train_fe_download"
                    )
                except Exception as e:
                    st.error(f"Download button failed: {e}")

        if 'test_fe' in st.session_state:
            test_csv, test_mime = get_download_file(st.session_state['test_fe'], "test_fe.csv")
            if test_csv and test_mime:
                try:
                    st.download_button(
                        label="Download Feature Engineered Test Data",
                        data=test_csv,
                        file_name="test_fe.csv",
                        mime=test_mime,
                        key="test_fe_download"
                    )
                except Exception as e:
                    st.error(f"Download button failed: {e}")

    st.divider()
    st.markdown("**Created with Belal Khamis, Marwa Kotb, Mahmoud Sabry, Mohamed Samy, Hoda Magdy**")

if __name__ == "__main__":
    main()
