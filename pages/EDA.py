import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram
from io import BytesIO
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Retail Sales Analysis</h1>", unsafe_allow_html=True)

def load_data(file_path, date_col, target_col=None):
    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df[['store_nbr', 'onpromotion']] = df[['store_nbr', 'onpromotion']].astype('int32')
    if target_col and target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col]).astype('float32')
    df.dropna(subset=[date_col], inplace=True)
    
    # Replace family categories
    family_map = {
        'AUTOMOTIVE': 'Tools', 'HARDWARE': 'Tools', 'LAWN AND GARDEN': 'Tools', 'PLAYERS AND ELECTRONICS': 'Tools',
        'BEAUTY': 'LifeStyle', 'LINGERIE': 'LifeStyle', 'LADIESWEAR': 'LifeStyle', 'PERSONAL CARE': 'LifeStyle',
        'CELEBRATION': 'LifeStyle', 'MAGAZINES': 'LifeStyle', 'BOOKS': 'LifeStyle', 'BABY CARE': 'LifeStyle',
        'HOME APPLIANCES': 'Home', 'HOME AND KITCHEN I': 'Home', 'HOME AND KITCHEN II': 'Home', 'HOME CARE': 'Home',
        'SCHOOL AND OFFICE SUPPLIES': 'Home', 'GROCERY II': 'Food', 'PET SUPPLIES': 'Food', 'SEAFOOD': 'Food',
        'LIQUOR,WINE,BEER': 'Food', 'DELI': 'Daily', 'EGGS': 'Daily'
    }
    df['family'] = df['family'].replace(family_map)
    
    # Create city_state and type_locale if source columns exist
    if 'city' in df.columns and 'state' in df.columns:
        df['city_state'] = df['city'] + "_" + df['state']
    if 'type_y' in df.columns and 'locale' in df.columns:
        df['type_locale'] = df['type_y'] + "_" + df['locale']
    
    return df

def plot_sales_trends(df, date_col, target_col, granularity='D'):
    sales_df = df.groupby(date_col)[target_col].sum().reset_index()
    sales_df.set_index(date_col, inplace=True)
    sales = sales_df.resample(granularity).sum().reset_index()
    fig = px.line(sales, x=date_col, y=target_col, title=f"Sales Trends ({granularity})")
    fig.update_layout(xaxis_title="Date", yaxis_title=target_col, xaxis_tickangle=45, yaxis_gridcolor='lightgray')
    return fig

def plot_sales_time_series(df, date_col, target_col):
    sales = df.groupby(date_col)[target_col].mean().reset_index()
    fig = px.line(sales, x=date_col, y=target_col, title="Sales Time Series")
    fig.update_layout(xaxis_title="Date", yaxis_title=target_col, xaxis_tickangle=45, yaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_family(df, target_col):
    sales = df.groupby('family')[target_col].mean().sort_values().reset_index()
    fig = px.bar(sales, y='family', x=target_col, orientation='h', title="Average Sales by Product Category")
    fig.update_layout(xaxis_title=target_col, yaxis_title="Family", yaxis_autorange='reversed', xaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_store(df, target_col):
    sales = df.groupby('store_nbr')[target_col].mean().sort_values().reset_index()
    fig = px.bar(sales, x='store_nbr', y=target_col, title="Average Sales by Store")
    fig.update_layout(xaxis_title="Store Number", yaxis_title=target_col, xaxis_tickangle=45, yaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_city_state(df, target_col):
    if 'city_state' in df.columns:
        sales = df.groupby('city_state')[target_col].mean().sort_values().reset_index()
        fig = px.bar(sales, y='city_state', x=target_col, orientation='h', title="Average Sales by City-State")
        fig.update_layout(xaxis_title=target_col, yaxis_title="City-State", yaxis_autorange='reversed', xaxis_gridcolor='lightgray')
        return fig
    return None

def plot_sales_by_type_locale(df, target_col):
    if 'type_locale' in df.columns:
        sales = df.groupby('type_locale')[target_col].mean().reset_index()
        fig = px.pie(sales, values=target_col, names='type_locale', title="Sales Distribution by Type-Locale")
        fig.update_layout(yaxis_gridcolor='lightgray')
        return fig
    return None

def plot_promotion_impact(df, target_col):
    fig = px.box(df, x='onpromotion', y=target_col, title="Sales Distribution by Promotion")
    fig.update_layout(xaxis_title="On Promotion", yaxis_title=target_col, xaxis_tickmode='linear', yaxis_gridcolor='lightgray')
    return fig

def plot_seasonal_decomposition(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    df_ts = df_agg.set_index(date_col)[target_col]
    decomp = seasonal_decompose(df_ts, model='additive', period=12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name="Trend", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name="Seasonal", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, name="Residual", line=dict(color="red")))
    fig.update_layout(title="Seasonal Decomposition of Sales", xaxis_title="Date", yaxis_title=target_col, yaxis_gridcolor='lightgray')
    return fig

def plot_rolling_stats(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    rolling = df_agg.set_index(date_col)[target_col].rolling(window=30).agg(['mean', 'std']).dropna().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling[date_col], y=rolling['mean'], name="Rolling Mean", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=rolling[date_col], y=rolling['std'], name="Rolling Std", line=dict(color="orange")))
    fig.update_layout(title="Rolling Mean and Std (30 Days)", xaxis_title="Date", yaxis_title=target_col, yaxis_gridcolor='lightgray')
    return fig

def plot_periodogram(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    freq, psd = periodogram(df_agg[target_col].dropna())
    fig = px.line(x=freq, y=psd, title="Periodogram of Sales")
    fig.update_layout(xaxis_title="Frequency", yaxis_title="Power Spectral Density", yaxis_gridcolor='lightgray')
    return fig

def plot_lag_plot(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    lag = 1
    df_agg[f'{target_col}_lag'] = df_agg[target_col].shift(lag)
    fig = px.scatter(df_agg, x=f'{target_col}_lag', y=target_col, title=f"Lag Plot (Lag={lag})")
    fig.update_layout(xaxis_title=f"{target_col} (t-{lag})", yaxis_title=target_col, yaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_dow(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    df_agg['dow'] = df_agg[date_col].dt.dayofweek
    sales_by_dow = df_agg.groupby('dow')[target_col].mean().reset_index()
    fig = px.line(sales_by_dow, x='dow', y=target_col, title="Average Sales by Day of Week")
    fig.update_layout(xaxis_title="Day of Week", yaxis_title=target_col, xaxis_tickmode='linear', yaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_month(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    df_agg['month'] = df_agg[date_col].dt.month
    sales_by_month = df_agg.groupby('month')[target_col].mean().reset_index()
    fig = px.line(sales_by_month, x='month', y=target_col, title="Average Sales by Month")
    fig.update_layout(xaxis_title="Month", yaxis_title=target_col, xaxis_tickmode='linear', yaxis_gridcolor='lightgray')
    return fig

def plot_correlation_heatmap(df, target_col):
    numeric_df = df.select_dtypes(include=['number']).corr()
    fig = go.Figure(data=go.Heatmap(
        z=numeric_df.values, x=numeric_df.columns, y=numeric_df.columns,
        colorscale='RdBu', text=numeric_df.round(2).values, texttemplate="%{text}"
    ))
    fig.update_layout(title="Correlation Heatmap", xaxis_title="Variables", yaxis_title="Variables", yaxis_gridcolor='lightgray')
    return fig

def plot_acf(df, date_col, target_col):
    series = df[(df['store_nbr'] == 1) & (df['family'] == 'GROCERY I')].groupby(date_col)[target_col].sum()
    if len(series) > 28 and not series.isna().all():
        acf_vals, acf_confint = acf(series.dropna(), nlags=28, alpha=0.05)
        lags = list(range(len(acf_vals)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lags, y=acf_vals, mode='markers+lines', name='ACF'))
        fig.add_trace(go.Scatter(x=lags, y=acf_confint[:, 0], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
        fig.add_trace(go.Scatter(x=lags, y=acf_confint[:, 1], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
        fig.add_hline(y=0, line=dict(color="black", dash="dash"))
        fig.update_layout(title='ACF for Store 1, GROCERY I', xaxis_title='Lag', yaxis_title='Autocorrelation', yaxis_gridcolor='lightgray')
        return fig
    return None

def plot_pacf(df, date_col, target_col):
    series = df[(df['store_nbr'] == 1) & (df['family'] == 'GROCERY I')].groupby(date_col)[target_col].sum()
    if len(series) > 28 and not series.isna().all():
        pacf_vals, pacf_confint = pacf(series.dropna(), nlags=28, alpha=0.05)
        lags = list(range(len(pacf_vals)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lags, y=pacf_vals, mode='markers+lines', name='PACF'))
        fig.add_trace(go.Scatter(x=lags, y=pacf_confint[:, 0], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
        fig.add_trace(go.Scatter(x=lags, y=pacf_confint[:, 1], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
        fig.add_hline(y=0, line=dict(color="black", dash="dash"))
        fig.update_layout(title='PACF for Store 1, GROCERY I', xaxis_title='Lag', yaxis_title='Partial Autocorrelation', yaxis_gridcolor='lightgray')
        return fig
    return None

def train_xgboost_model(df, target_col):
    df = df.copy()
    # Identify available categorical columns
    categorical_cols = ['date', 'family', 'city_state', 'type_locale']
    available_cat_cols = [col for col in categorical_cols if col in df.columns]
    for col in available_cat_cols:
        df[col] = df[col].astype('category').cat.add_categories(['Unknown']).fillna('Unknown')
    
    # Identify and preprocess numerical columns
    numerical_cols = [col for col in df.columns if col not in available_cat_cols + [target_col]]
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Define pipeline with available categorical columns
    pipeline = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), available_cat_cols)
    ], remainder='passthrough')
    
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    model = XGBRegressor(n_estimators=10, max_depth=20, verbosity=2)
    model.fit(X_train_transformed, y_train)
    
    score = model.score(X_test_transformed, y_test)
    return model, pipeline, score

def predict_sales(model, encoder, test_df):
    test_df = test_df.copy()
    categorical_cols = ['date', 'family', 'city_state', 'type_locale']
    available_cat_cols = [col for col in categorical_cols if col in test_df.columns]
    for col in available_cat_cols:
        test_df[col] = test_df[col].astype('category').cat.add_categories(['Unknown']).fillna('Unknown')
    
    numerical_cols = [col for col in test_df.columns if col not in available_cat_cols]
    for col in numerical_cols:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        test_df[col] = test_df[col].fillna(test_df[col].mean())
    
    X_test = encoder.transform(test_df)
    predictions = model.predict(X_test)
    return predictions

def get_download_file(df, filename):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue(), 'text/csv'

def main():
    train_tab, test_tab = st.tabs(["Train", "Test"])

    with train_tab:
        train_file = st.file_uploader("Upload Train Data (.csv)", ['csv'], key="train")
        if train_file:
            st.session_state['train_content'] = train_file.getvalue()
            st.session_state['train_file'] = train_file
            train = load_data(BytesIO(st.session_state['train_content']), 'date', 'sales')
            if 'Unnamed: 17' in train.columns:
                train.drop('Unnamed: 17', axis=1, inplace=True)
            with st.form("train_config"):
                st.dataframe(train.head())
                date_col = st.selectbox("Date Column", train.columns, index=train.columns.get_loc('date') if 'date' in train.columns else 0)
                target_col = st.selectbox("Target Column", train.columns, index=train.columns.get_loc('sales') if 'sales' in train.columns else 0)
                submitted = st.form_submit_button("Apply")
                if submitted:
                    st.session_state['train_df'] = train
                    st.session_state['train_date'] = date_col
                    st.session_state['train_target'] = target_col
                    st.session_state['train_configured'] = True

        if st.session_state.get('train_configured'):
            train = st.session_state['train_df']
            if st.button("Generate Plots", key="train_plots"):
                plots = [
                    plot_sales_time_series(train, st.session_state['train_date'], st.session_state['train_target']),
                    plot_sales_trends(train, st.session_state['train_date'], st.session_state['train_target'], 'D'),
                    plot_sales_trends(train, st.session_state['train_date'], st.session_state['train_target'], 'W'),
                    plot_sales_by_family(train, st.session_state['train_target']),
                    plot_sales_by_store(train, st.session_state['train_target']),
                    plot_sales_by_city_state(train, st.session_state['train_target']),
                    plot_sales_by_type_locale(train, st.session_state['train_target']),
                    plot_promotion_impact(train, st.session_state['train_target']),
                    plot_seasonal_decomposition(train, st.session_state['train_date'], st.session_state['train_target']),
                    plot_rolling_stats(train, st.session_state['train_date'], st.session_state['train_target']),
                    plot_periodogram(train, st.session_state['train_date'], st.session_state['train_target']),
                    plot_lag_plot(train, st.session_state['train_date'], st.session_state['train_target']),
                    plot_sales_by_dow(train, st.session_state['train_date'], st.session_state['train_target']),
                    plot_sales_by_month(train, st.session_state['train_date'], st.session_state['train_target']),
                    plot_correlation_heatmap(train, st.session_state['train_target']),
                    plot_acf(train, st.session_state['train_date'], st.session_state['train_target']),
                    plot_pacf(train, st.session_state['train_date'], st.session_state['train_target'])
                ]
                
                for plot in plots:
                    if plot:
                        st.plotly_chart(plot)
                
                model, encoder, score = train_xgboost_model(train, st.session_state['train_target'])
                st.write(f"XGBoost Model R² Score: {score:.4f}")
                st.session_state['model'] = model
                st.session_state['encoder'] = encoder
                
                st.download_button(
                    label="Download Train Data",
                    data=get_download_file(train, "train_data.csv")[0],
                    file_name="train_data.csv",
                    mime="text/csv"
                )

    with test_tab:
        test_file = st.file_uploader("Upload Test Data (.csv)", ['csv'], key="test")
        if test_file:
            st.session_state['test_content'] = test_file.getvalue()
            st.session_state['test_file'] = test_file
            test = load_data(BytesIO(st.session_state['test_content']), 'date', target_col=None)
            with st.form("test_config"):
                st.dataframe(test.head())
                date_col = st.selectbox("Date Column", test.columns, index=test.columns.get_loc('date') if 'date' in test.columns else 0)
                target_col_options = ['None'] + list(test.columns)
                target_col = st.selectbox("Target Column (optional)", target_col_options, index=0)
                submitted = st.form_submit_button("Apply")
                if submitted:
                    st.session_state['test_df'] = test
                    st.session_state['test_date'] = date_col
                    st.session_state['test_target'] = None if target_col == 'None' else target_col
                    st.session_state['test_configured'] = True

        if st.session_state.get('test_configured') and st.button("Generate Plots", key="test_plots"):
            test = st.session_state['test_df']
            target_col = st.session_state['test_target'] or 'onpromotion'
            
            if st.session_state.get('model') and st.session_state.get('encoder'):
                predictions = predict_sales(st.session_state['model'], st.session_state['encoder'], test)
                test['predicted_sales'] = predictions
                target_col = 'predicted_sales'
            
            plots = [
                plot_sales_time_series(test, st.session_state['test_date'], target_col),
                plot_sales_trends(test, st.session_state['test_date'], target_col, 'D'),
                plot_sales_trends(test, st.session_state['test_date'], target_col, 'W'),
                plot_sales_by_family(test, target_col),
                plot_sales_by_store(test, target_col),
                plot_sales_by_city_state(test, target_col),
                plot_sales_by_type_locale(test, target_col),
                plot_promotion_impact(test, target_col)
            ]
            
            for plot in plots:
                if plot:
                    st.plotly_chart(plot)
            
            st.download_button(
                label="Download Test Data",
                data=get_download_file(test, "test_data.csv")[0],
                file_name="test_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
