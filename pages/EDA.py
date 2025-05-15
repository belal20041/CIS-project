import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram
from io import BytesIO

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Retail Sales Analysis</h1>", unsafe_allow_html=True)

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

def plot_sales_by_family(df, target_col):
    family_map = {'AUTOMOTIVE': 'Tools', 'HARDWARE': 'Tools', 'LAWN AND GARDEN': 'Tools', 'PLAYERS AND ELECTRONICS': 'Tools', 'BEAUTY': 'LifeStyle', 'LINGERIE': 'LifeStyle', 'LADIESWEAR': 'LifeStyle', 'PERSONAL CARE': 'LifeStyle', 'CELEBRATION': 'LifeStyle', 'MAGAZINES': 'LifeStyle', 'BOOKS': 'LifeStyle', 'BABY CARE': 'LifeStyle', 'HOME APPLIANCES': 'Home', 'HOME AND KITCHEN I': 'Home', 'HOME AND KITCHEN II': 'Home', 'HOME CARE': 'Home', 'SCHOOL AND OFFICE SUPPLIES': 'Home', 'GROCERY II': 'Food', 'PET SUPPLIES': 'Food', 'SEAFOOD': 'Food', 'LIQUOR,WINE,BEER': 'Food', 'DELI': 'Daily', 'EGGS': 'Daily'}
    df['family'] = df['family'].replace(family_map)
    sales = df.groupby('family')[target_col].mean().sort_values().reset_index()
    fig = px.bar(sales, y='family', x=target_col, orientation='h', title="Average Sales by Product Category")
    fig.update_layout(yaxis_autorange='reversed', xaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_store(df, target_col):
    sales = df.groupby('store_nbr')[target_col].mean().sort_values().reset_index()
    fig = px.bar(sales, x='store_nbr', y=target_col, title="Average Sales by Store")
    fig.update_layout(xaxis_tickangle=45, yaxis_gridcolor='lightgray')
    return fig

def plot_promotion_impact(df, target_col):
    fig = px.box(df, x='onpromotion', y=target_col, title="Sales Distribution by Promotion")
    fig.update_layout(xaxis_tickmode='linear', yaxis_gridcolor='lightgray')
    return fig

def plot_seasonal_decomposition(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    df_ts = df_agg.set_index(date_col)[target_col].resample('M').sum()
    decomp = seasonal_decompose(df_ts, model='additive', period=12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name="Trend", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name="Seasonal", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, name="Residual", line=dict(color="red")))
    fig.update_layout(title="Seasonal Decomposition of Sales", xaxis_title="Date", yaxis_title="Sales", yaxis_gridcolor='lightgray')
    return fig

def plot_rolling_stats(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    rolling = df_agg.set_index(date_col)[target_col].rolling(window=30).agg(['mean', 'std']).dropna().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling[date_col], y=rolling['mean'], name="Rolling Mean", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=rolling[date_col], y=rolling['std'], name="Rolling Std", line=dict(color="orange")))
    fig.update_layout(title="Rolling Mean and Std (30 Days)", xaxis_title="Date", yaxis_title="Sales", yaxis_gridcolor='lightgray')
    return fig

def plot_periodogram(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    freq, psd = periodogram(df_agg[target_col].dropna())
    fig = px.line(x=freq, y=psd, title="Periodogram of Sales")
    fig.update_layout(yaxis_gridcolor='lightgray')
    return fig

def plot_lag_plot(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    lag = 1
    df_agg[f'{target_col}_lag'] = df_agg[target_col].shift(lag)
    fig = px.scatter(df_agg, x=f'{target_col}_lag', y=target_col, title=f"Lag Plot (Lag={lag})")
    fig.update_layout(yaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_dow(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    df_agg['dow'] = df_agg[date_col].dt.dayofweek
    sales_by_dow = df_agg.groupby('dow')[target_col].mean().reset_index()
    fig = px.line(sales_by_dow, x='dow', y=target_col, title="Average Sales by Day of Week")
    fig.update_layout(xaxis_tickmode='linear', yaxis_gridcolor='lightgray')
    return fig

def plot_sales_by_month(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    df_agg['month'] = df_agg[date_col].dt.month
    sales_by_month = df_agg.groupby('month')[target_col].mean().reset_index()
    fig = px.line(sales_by_month, x='month', y=target_col, title="Average Sales by Month")
    fig.update_layout(xaxis_tickmode='linear', yaxis_gridcolor='lightgray')
    return fig

def plot_correlation_heatmap(df, target_col):
    numeric_df = df.select_dtypes(include=['number']).corr()
    fig = go.Figure(data=go.Heatmap(z=numeric_df.values, x=numeric_df.columns, y=numeric_df.columns, colorscale='RdBu', text=numeric_df.round(2).values, texttemplate="%{text}"))
    fig.update_layout(title="Correlation Heatmap", xaxis_title="Variables", yaxis_title="Variables", yaxis_gridcolor='lightgray')
    return fig

def plot_acf(df, date_col, target_col):
    series = df[(df['store_nbr'] == 1) & (df['family'] == 'GROCERY I')].groupby(date_col)[target_col].sum()
    if len(series) > 28:
        acf_vals, acf_confint = acf(series.dropna(), nlags=28, alpha=0.05)
        lags = range(len(acf_vals))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lags, y=acf_vals, mode='markers+lines', name='ACF'))
        fig.add_trace(go.Scatter(x=lags, y=acf_confint[:, 0], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
        fig.add_trace(go.Scatter(x=lags, y=acf_confint[:, 1], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
        fig.add_hline(y=0, line=dict(color="black", dash="dash"))
        fig.update_layout(title='ACF for Store 1, GROCERY I', xaxis_title='Lag', yaxis_title='Autocorrelation', yaxis_gridcolor='lightgray')
        return fig

def plot_pacf(df, date_col, target_col):
    series = df[(df['store_nbr'] == 1) & (df['family'] == 'GROCERY I')].groupby(date_col)[target_col].sum()
    if len(series) > 28:
        pacf_vals, pacf_confint = pacf(series.dropna(), nlags=28, alpha=0.05)
        lags = range(len(pacf_vals))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lags, y=pacf_vals, mode='markers+lines', name='PACF'))
        fig.add_trace(go.Scatter(x=lags, y=pacf_confint[:, 0], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
        fig.add_trace(go.Scatter(x=lags, y=pacf_confint[:, 1], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)'))
        fig.add_hline(y=0, line=dict(color="black", dash="dash"))
        fig.update_layout(title='PACF for Store 1, GROCERY I', xaxis_title='Lag', yaxis_title='Partial Autocorrelation', yaxis_gridcolor='lightgray')
        return fig

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
            st.session_state['train_content'] = train_file.read()
            st.session_state['train_file'] = train_file
            train = load_data(BytesIO(st.session_state['train_content']), 'date', 'sales')
            with st.form("train_config"):
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
                fig3 = plot_sales_by_family(train, st.session_state['train_target'])
                fig4 = plot_sales_by_store(train, st.session_state['train_target'])
                fig5 = plot_promotion_impact(train, st.session_state['train_target'])
                fig6 = plot_seasonal_decomposition(train, st.session_state['train_date'], st.session_state['train_target'])
                fig7 = plot_rolling_stats(train, st.session_state['train_date'], st.session_state['train_target'])
                fig8 = plot_periodogram(train, st.session_state['train_date'], st.session_state['train_target'])
                fig9 = plot_lag_plot(train, st.session_state['train_date'], st.session_state['train_target'])
                fig10 = plot_sales_by_dow(train, st.session_state['train_date'], st.session_state['train_target'])
                fig11 = plot_sales_by_month(train, st.session_state['train_date'], st.session_state['train_target'])
                fig12 = plot_correlation_heatmap(train, st.session_state['train_target'])
                fig13 = plot_acf(train, st.session_state['train_date'], st.session_state['train_target'])
                fig14 = plot_pacf(train, st.session_state['train_date'], st.session_state['train_target'])
                st.plotly_chart(fig1)
                st.plotly_chart(fig2)
                st.plotly_chart(fig3)
                st.plotly_chart(fig4)
                st.plotly_chart(fig5)
                st.plotly_chart(fig6)
                st.plotly_chart(fig7)
                st.plotly_chart(fig8)
                st.plotly_chart(fig9)
                st.plotly_chart(fig10)
                st.plotly_chart(fig11)
                st.plotly_chart(fig12)
                if fig13: st.plotly_chart(fig13)
                if fig14: st.plotly_chart(fig14)
                st.download_button("Download Train Data", get_download_file(train, "train_data.csv")[0], "train_data.csv", "text/csv")

    with test_tab:
        test_file = st.file_uploader("Upload Test Data (.csv)", ['csv'], key="test")
        if test_file:
            st.session_state['test_content'] = test_file.read()
            st.session_state['test_file'] = test_file
            test = load_data(BytesIO(st.session_state['test_content']), 'date', 'sales')
            with st.form("test_config"):
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
                fig3 = plot_sales_by_family(test, 'sales')
                fig4 = plot_sales_by_store(test, 'sales')
                fig5 = plot_promotion_impact(test, 'sales')
                st.plotly_chart(fig1)
                st.plotly_chart(fig2)
                st.plotly_chart(fig3)
                st.plotly_chart(fig4)
                st.plotly_chart(fig5)
                st.download_button("Download Test Data", get_download_file(test, "test_data.csv")[0], "test_data.csv", "text/csv")

if __name__ == "__main__":
    main()
