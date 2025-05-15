import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import periodogram
from io import BytesIO
import hashlib

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Retail Sales Time Series Analysis</h1>", unsafe_allow_html=True)

TRAIN_END = "2017-07-15"
VAL_END = "2017-08-15"

def hash_content(content):
    return hashlib.sha256(content).hexdigest()

def load_data(file_path, date_col, target_col):
    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df[['store_nbr', 'onpromotion']] = df[['store_nbr', 'onpromotion']].astype('int32')
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col]).astype('float32')
    df.dropna(subset=[date_col], inplace=True)
    return df

def prepare_data(train, test, date_col, target_col):
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test]).sort_values([date_col, 'store_nbr', 'family'])
    agg_dict = {target_col: 'sum', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first'}
    combined = combined.groupby([date_col, 'store_nbr', 'family']).agg(agg_dict).reset_index()
    combined = combined.astype({date_col: 'datetime64[ns]', target_col: 'float32', 'onpromotion': 'int32', 'is_train': 'int8', 'store_nbr': 'int32', 'family': 'category'})
    return combined

def fill_missing(df, target_col):
    grouped = df.groupby(['store_nbr', 'family'])
    result = []
    for _, group in grouped:
        group[target_col] = group[target_col].ffill().fillna(0).astype('float32')
        group['onpromotion'] = group['onpromotion'].fillna(0).astype('int32')
        result.append(group)
    return pd.concat(result)

def add_features(df, date_col, target_col):
    df['day'] = df[date_col].dt.day.astype('int8')
    df['dow'] = df[date_col].dt.dayofweek.astype('int8')
    df['month'] = df[date_col].dt.month.astype('int8')
    df['year'] = df[date_col].dt.year.astype('int16')
    for lag in [7, 14]:
        df[f'lag_{lag}'] = df.groupby(['store_nbr', 'family'])[target_col].shift(lag).astype('float32')
    df['roll_mean_7'] = df.groupby(['store_nbr', 'family'])[target_col].shift(1).rolling(7).mean().astype('float32')
    return df

def split_data(df, date_col, target_col):
    train = df[df['is_train'] == 1]
    test = df[df['is_train'] == 0].drop(target_col, axis=1)
    train_set = train[train[date_col] <= TRAIN_END]
    val_set = train[(train[date_col] > TRAIN_END) & (train[date_col] <= VAL_END)]
    return train_set, val_set, test

def get_download_file(df, filename):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue(), 'text/csv'

def reclassify_family(df):
    family_map = {'AUTOMOTIVE': 'Tools', 'HARDWARE': 'Tools', 'LAWN AND GARDEN': 'Tools', 'PLAYERS AND ELECTRONICS': 'Tools', 'BEAUTY': 'LifeStyle', 'LINGERIE': 'LifeStyle', 'LADIESWEAR': 'LifeStyle', 'PERSONAL CARE': 'LifeStyle', 'CELEBRATION': 'LifeStyle', 'MAGAZINES': 'LifeStyle', 'BOOKS': 'LifeStyle', 'BABY CARE': 'LifeStyle', 'HOME APPLIANCES': 'Home', 'HOME AND KITCHEN I': 'Home', 'HOME AND KITCHEN II': 'Home', 'HOME CARE': 'Home', 'SCHOOL AND OFFICE SUPPLIES': 'Home', 'GROCERY II': 'Food', 'PET SUPPLIES': 'Food', 'SEAFOOD': 'Food', 'LIQUOR,WINE,BEER': 'Food', 'DELI': 'Daily', 'EGGS': 'Daily'}
    df['family'] = df['family'].replace(family_map)
    return df

def plot_sales_trends(df, date_col, granularity='D', family_filter=None):
    if family_filter:
        df = df[df['family'] == family_filter]
    sales = df.groupby(pd.Grouper(key=date_col, freq=granularity))['sales'].sum().reset_index()
    fig = px.line(sales, x=date_col, y='sales', title=f"Sales Trends ({granularity})", labels={'sales': 'Total Sales', date_col: 'Date'})
    holidays = df[df['type_y'] == 'Holiday'][date_col].unique()
    for holiday in holidays:
        holiday_str = holiday.strftime('%Y-%m-%d')
        fig.add_vline(x=holiday_str, line=dict(color="red", dash="dash"))
    fig.update_layout(xaxis_tickangle=45, yaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_family(df):
    df = reclassify_family(df)
    sales = df.groupby('family')['sales'].mean().sort_values().reset_index()
    fig = px.bar(sales, y='family', x='sales', orientation='h', title="Average Sales by Product Category", labels={'sales': 'Avg Sales', 'family': 'Category'})
    fig.update_layout(yaxis_autorange='reversed', xaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_store(df):
    sales = df.groupby('store_nbr')['sales'].mean().sort_values().reset_index()
    fig = px.bar(sales, x='store_nbr', y='sales', title="Average Sales by Store", labels={'sales': 'Avg Sales', 'store_nbr': 'Store Number'})
    fig.update_layout(xaxis_tickangle=45, yaxis_gridcolor='lightgray')
    return fig

def plot_promotion_impact(df):
    fig = px.box(df, x='onpromotion', y='sales', title="Sales Distribution by Promotion", labels={'sales': 'Sales', 'onpromotion': 'On Promotion (0 = No, 1 = Yes)'})
    fig.update_layout(xaxis_tickmode='linear', yaxis_gridcolor='lightgray')
    return fig

def plot_seasonal_decomposition(df, date_col):
    df_ts = df.set_index(date_col)['sales'].resample('M').sum()
    decomp = seasonal_decompose(df_ts, model='additive', period=12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name="Trend", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name="Seasonal", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, name="Residual", line=dict(color="red")))
    fig.update_layout(title="Seasonal Decomposition of Sales", xaxis_title="Date", yaxis_title="Sales", yaxis_gridcolor='lightgray')
    return fig

def plot_autocorrelation(df, target_col):
    acf_vals, _ = acf(df[target_col].dropna(), nlags=30, fft=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF", marker_color='blue'))
    fig.update_layout(title="Autocorrelation (ACF) of Sales", xaxis_title="Lag", yaxis_title="Autocorrelation", yaxis_gridcolor='lightgray')
    return fig

def plot_partial_autocorrelation(df, target_col):
    pacf_vals, _ = pacf(df[target_col].dropna(), nlags=30)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF", marker_color='blue'))
    fig.update_layout(title="Partial Autocorrelation (PACF) of Sales", xaxis_title="Lag", yaxis_title="Partial Autocorrelation", yaxis_gridcolor='lightgray')
    return fig

def plot_rolling_stats(df, date_col, target_col):
    rolling = df.set_index(date_col)[target_col].rolling(window=30).agg(['mean', 'std']).dropna().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling[date_col], y=rolling['mean'], name="Rolling Mean", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=rolling[date_col], y=rolling['std'], name="Rolling Std", line=dict(color="orange")))
    fig.update_layout(title="Rolling Mean and Std (30 Days)", xaxis_title="Date", yaxis_title="Sales", yaxis_gridcolor='lightgray')
    return fig

def plot_periodogram(df, target_col):
    freq, psd = periodogram(df[target_col].dropna())
    fig = px.line(x=freq, y=psd, title="Periodogram of Sales", labels={'x': 'Frequency', 'y': 'Power Spectral Density'})
    fig.update_layout(yaxis_gridcolor='lightgray')
    return fig

def plot_lag_plot(df, target_col):
    lag = 1
    df[f'{target_col}_lag'] = df[target_col].shift(lag)
    fig = px.scatter(df, x=f'{target_col}_lag', y=target_col, title=f"Lag Plot (Lag={lag})", labels={f'{target_col}_lag': f'Sales (t-{lag})', target_col: 'Sales (t)'})
    fig.update_layout(yaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_dow(df, date_col, target_col):
    df['dow'] = df[date_col].dt.dayofweek
    sales_by_dow = df.groupby('dow')[target_col].mean().reset_index()
    fig = px.line(sales_by_dow, x='dow', y=target_col, title="Average Sales by Day of Week", labels={'dow': 'Day of Week (0=Mon, 6=Sun)', target_col: 'Average Sales'})
    fig.update_layout(xaxis_tickmode='linear', yaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_month(df, date_col, target_col):
    df['month'] = df[date_col].dt.month
    sales_by_month = df.groupby('month')[target_col].mean().reset_index()
    fig = px.line(sales_by_month, x='month', y=target_col, title="Average Sales by Month", labels={'month': 'Month', target_col: 'Average Sales'})
    fig.update_layout(xaxis_tickmode='linear', yaxis_gridcolor='lightgray')
    return fig

def main():
    train_tab, test_tab = st.tabs(["Train", "Test"])

    with train_tab:
        train_file = st.file_uploader("Upload Train Data (.csv)", ['csv'], key="train")
        if train_file:
            if 'train_content' not in st.session_state or st.session_state['train_file'] != train_file:
                content = train_file.read()
                st.session_state['train_content'] = content
                st.session_state['train_hash'] = hash_content(content)
                st.session_state['train_file'] = train_file

            with st.form("train_config"):
                train = load_data(BytesIO(st.session_state['train_content']), 'date', 'sales')
                st.dataframe(train.head())
                date_col = st.selectbox("Date Column", train.columns, index=train.columns.tolist().index('date') if 'date' in train.columns else 0)
                target_col = st.selectbox("Target Column", train.columns, index=train.columns.tolist().index('sales') if 'sales' in train.columns else 0)
                if st.form_submit_button("Apply"):
                    st.session_state['train_df'] = train
                    st.session_state['train_date'] = date_col
                    st.session_state['train_target'] = target_col
                    st.session_state['configured'] = True

            if 'configured' in st.session_state:
                train = st.session_state['train_df']
                st.dataframe(train.head())

                if st.button("Generate Time Series Plots"):
                    fig1 = plot_sales_trends(train, st.session_state['train_date'], 'D')
                    fig2 = plot_sales_trends(train, st.session_state['train_date'], 'W')
                    fig3 = plot_sales_by_family(train)
                    fig4 = plot_sales_by_store(train)
                    fig5 = plot_promotion_impact(train)
                    fig6 = plot_seasonal_decomposition(train, st.session_state['train_date'])
                    fig7 = plot_autocorrelation(train, st.session_state['train_target'])
                    fig8 = plot_partial_autocorrelation(train, st.session_state['train_target'])
                    fig9 = plot_rolling_stats(train, st.session_state['train_date'], st.session_state['train_target'])
                    fig10 = plot_periodogram(train, st.session_state['train_target'])
                    fig11 = plot_lag_plot(train, st.session_state['train_target'])
                    fig12 = plot_sales_by_dow(train, st.session_state['train_date'], st.session_state['train_target'])
                    fig13 = plot_sales_by_month(train, st.session_state['train_date'], st.session_state['train_target'])

                    st.plotly_chart(fig1)
                    st.plotly_chart(fig2)
                    st.plotly_chart(fig3)
                    st.plotly_chart(fig4)
                    st.plotly_chart(fig5)
                    if fig6:
                        st.plotly_chart(fig6)
                    st.plotly_chart(fig7)
                    st.plotly_chart(fig8)
                    st.plotly_chart(fig9)
                    st.plotly_chart(fig10)
                    st.plotly_chart(fig11)
                    st.plotly_chart(fig12)
                    st.plotly_chart(fig13)

                    data = get_download_file(train, "train_data.csv")
                    st.download_button("Download Train Data", data[0], "train_data.csv", data[1])

    with test_tab:
        test_file = st.file_uploader("Upload Test Data (.csv)", ['csv'], key="test")
        if test_file:
            if 'test_content' not in st.session_state or st.session_state['test_file'] != test_file:
                content = test_file.read()
                st.session_state['test_content'] = content
                st.session_state['test_hash'] = hash_content(content)
                st.session_state['test_file'] = test_file

            with st.form("test_config"):
                test = load_data(BytesIO(st.session_state['test_content']), 'date', None)
                st.dataframe(test.head())
                date_col = st.selectbox("Date Column", test.columns, index=test.columns.tolist().index('date') if 'date' in test.columns else 0)
                if st.form_submit_button("Apply"):
                    st.session_state['test_df'] = test
                    st.session_state['test_date'] = date_col
                    st.session_state['configured'] = True

            if 'configured' in st.session_state:
                test = st.session_state['test_df']
                st.dataframe(test.head())

                if st.button("Generate Time Series Plots"):
                    fig1 = plot_sales_trends(test, st.session_state['test_date'], 'D')
                    fig2 = plot_sales_trends(test, st.session_state['test_date'], 'W')
                    fig3 = plot_sales_by_family(test)
                    fig4 = plot_sales_by_store(test)
                    fig5 = plot_promotion_impact(test)

                    st.plotly_chart(fig1)
                    st.plotly_chart(fig2)
                    st.plotly_chart(fig3)
                    st.plotly_chart(fig4)
                    st.plotly_chart(fig5)

                    data = get_download_file(test, "test_data.csv")
                    st.download_button("Download Test Data", data[0], "test_data.csv", data[1])

    if 'train_df' in st.session_state and 'test_df' in st.session_state:
        with st.form("features"):
            if st.form_submit_button("Generate Features"):
                combined = prepare_data(st.session_state['train_df'], st.session_state['test_df'], st.session_state['train_date'], st.session_state['train_target'])
                combined = fill_missing(combined, st.session_state['train_target'])
                combined = add_features(combined, st.session_state['train_date'], st.session_state['train_target'])
                train_set, val_set, test_set = split_data(combined, st.session_state['train_date'], st.session_state['train_target'])
                st.dataframe(train_set.head())
                st.dataframe(val_set.head())
                st.dataframe(test_set.head())

                data1 = get_download_file(train_set, "train_features.csv")
                st.download_button("Train Features", data1[0], "train_features.csv", data1[1])
                data2 = get_download_file(val_set, "val_features.csv")
                st.download_button("Val Features", data2[0], "val_features.csv", data2[1])
                data3 = get_download_file(test_set, "test_features.csv")
                st.download_button("Test Features", data3[0], "test_features.csv", data3[1])

    st.markdown("**Developed by Belal Khamis, Marwa Kotb, Mahmoud Sabry, Mohamed Samy, Hoda Magdy**")

if __name__ == "__main__":
    main()
