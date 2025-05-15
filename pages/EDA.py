import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import acf, pacf
from io import BytesIO
import hashlib

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Retail Sales Analysis</h1>", unsafe_allow_html=True)

def hash_content(content):
    return hashlib.sha256(content).hexdigest()

def load_data(file_path, date_col, target_col):
    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df[['store_nbr', 'onpromotion']] = df[['store_nbr', 'onpromotion']].astype('int32')
    df[target_col] = pd.to_numeric(df[target_col]).astype('float32')
    df.dropna(subset=[date_col], inplace=True)
    return df

def plot_sales_trends(df, date_col, target_col, granularity='D'):
    sales = df.groupby(pd.Grouper(key=date_col, freq=granularity))[target_col].sum().reset_index()
    fig = px.line(sales, x=date_col, y=target_col, title=f"Sales Trends ({granularity})")
    fig.update_layout(xaxis_tickangle=45, yaxis_gridcolor='lightgray')
    return fig

def plot_correlation_heatmap(df, target_col):
    numeric_df = df.select_dtypes(include=['number']).corr()
    fig = go.Figure(data=go.Heatmap(z=numeric_df.values, x=numeric_df.columns, y=numeric_df.columns, colorscale='RdBu', text=numeric_df.round(2).values, texttemplate="%{text}"))
    fig.update_layout(title="Correlation Heatmap", xaxis_title="Variables", yaxis_title="Variables", yaxis_gridcolor='lightgray')
    return fig

def plot_acf(df, date_col, target_col):
    series = df[(df['store_nbr'] == 1) & (df['family'] == 'GROCERY I')].sort_values(date_col).set_index(date_col)[target_col]
    n_lags = 28
    acf_vals, acf_confint = acf(series.dropna(), nlags=n_lags, alpha=0.05)
    lags = range(len(acf_vals))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags, y=acf_vals, mode='markers+lines', name='ACF'))
    fig.add_trace(go.Scatter(x=lags, y=acf_confint[:, 0], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
    fig.add_trace(go.Scatter(x=lags, y=acf_confint[:, 1], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig.update_layout(title='ACF for Store 1, GROCERY I', xaxis_title='Lag', yaxis_title='Autocorrelation', yaxis_gridcolor='lightgray')
    return fig

def plot_pacf(df, date_col, target_col):
    series = df[(df['store_nbr'] == 1) & (df['family'] == 'GROCERY I')].sort_values(date_col).set_index(date_col)[target_col]
    n_lags = 28
    pacf_vals, pacf_confint = pacf(series.dropna(), nlags=n_lags, alpha=0.05)
    lags = range(len(pacf_vals))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags, y=pacf_vals, mode='markers+lines', name='PACF'))
    fig.add_trace(go.Scatter(x=lags, y=pacf_confint[:, 0], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
    fig.add_trace(go.Scatter(x=lags, y=pacf_confint[:, 1], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig.update_layout(title='PACF for Store 1, GROCERY I', xaxis_title='Lag', yaxis_title='Partial Autocorrelation', yaxis_gridcolor='lightgray')
    return fig

def main():
    train_tab, test_tab = st.tabs(["Train", "Test"])

    with train_tab:
        train_file = st.file_uploader("Upload Train Data (.csv)", ['csv'], key="train")
        if train_file:
            if 'train_content' not in st.session_state or st.session_state['train_file'] != train_file:
                st.session_state['train_content'] = train_file.read()
                st.session_state['train_hash'] = hash_content(st.session_state['train_content'])
                st.session_state['train_file'] = train_file

            with st.form("train_config"):
                train = load_data(BytesIO(st.session_state['train_content']), 'date', 'sales')
                st.dataframe(train.head())
                date_col = st.selectbox("Date Column", train.columns, index=0)
                target_col = st.selectbox("Target Column", train.columns, index=0)
                if st.form_submit_button("Apply"):
                    st.session_state['train_df'] = train
                    st.session_state['train_date'] = date_col
                    st.session_state['train_target'] = target_col
                    st.session_state['configured'] = True

            if 'configured' in st.session_state:
                train = st.session_state['train_df']
                if st.button("Generate Plots"):
                    fig1 = plot_sales_trends(train, st.session_state['train_date'], st.session_state['train_target'], 'D')
                    fig2 = plot_sales_trends(train, st.session_state['train_date'], st.session_state['train_target'], 'W')
                    fig3 = plot_correlation_heatmap(train, st.session_state['train_target'])
                    fig4 = plot_acf(train, st.session_state['train_date'], st.session_state['train_target'])
                    fig5 = plot_pacf(train, st.session_state['train_date'], st.session_state['train_target'])
                    st.plotly_chart(fig1)
                    st.plotly_chart(fig2)
                    st.plotly_chart(fig3)
                    st.plotly_chart(fig4)
                    st.plotly_chart(fig5)
                    st.download_button("Download Train Data", get_download_file(train, "train_data.csv")[0], "train_data.csv", "text/csv")

    with test_tab:
        test_file = st.file_uploader("Upload Test Data (.csv)", ['csv'], key="test")
        if test_file:
            if 'test_content' not in st.session_state or st.session_state['test_file'] != test_file:
                st.session_state['test_content'] = test_file.read()
                st.session_state['test_hash'] = hash_content(st.session_state['test_content'])
                st.session_state['test_file'] = test_file

            with st.form("test_config"):
                test = load_data(BytesIO(st.session_state['test_content']), 'date', 'sales')
                st.dataframe(test.head())
                date_col = st.selectbox("Date Column", test.columns, index=0)
                if st.form_submit_button("Apply"):
                    st.session_state['test_df'] = test
                    st.session_state['test_date'] = date_col
                    st.session_state['configured'] = True

            if 'configured' in st.session_state:
                test = st.session_state['test_df']
                if st.button("Generate Plots"):
                    fig1 = plot_sales_trends(test, st.session_state['test_date'], 'sales', 'D')
                    fig2 = plot_sales_trends(test, st.session_state['test_date'], 'sales', 'W')
                    st.plotly_chart(fig1)
                    st.plotly_chart(fig2)
                    st.download_button("Download Test Data", get_download_file(test, "test_data.csv")[0], "test_data.csv", "text/csv")

if __name__ == "__main__":
    main()
