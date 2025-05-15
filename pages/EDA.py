import streamlit as st
import pandas as pd
import numpy as np
try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    st.error("The 'plotly' package is not installed. Please install it by running 'pip install plotly' or add it to your requirements.txt file.")
    st.stop()
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import periodogram
from io import BytesIO
import hashlib

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Retail Sales Time Series Analysis</h1>", unsafe_allow_html=True)

TRAIN_END = "2017-07-15"
VAL_END = "2017-08-15"

# Helper function to hash file content for caching
def hash_file(file):
    if hasattr(file, 'seek'):
        file.seek(0)
    content = file.read()
    file.seek(0)  # Reset file pointer after reading
    return hashlib.sha256(content).hexdigest()

@st.cache_data
def detect_column_types(df, date_col):
    numeric_cols = df.select_dtypes(['int64', 'float64']).columns.tolist()
    if date_col in numeric_cols:
        numeric_cols.remove(date_col)
    categorical_cols = [col for col in df.columns if col != date_col and 
                       (df[col].dtype in ['object', 'category', 'bool'] or df[col].nunique() / len(df) < 0.05)]
    return numeric_cols, categorical_cols

@st.cache_data
def load_data(file_content, file_hash, date_col, target_col):
    df = pd.read_csv(BytesIO(file_content))
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[['store_nbr', 'onpromotion']] = df[['store_nbr', 'onpromotion']].astype('int32')
    if target_col and target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').astype('float32')
    df.dropna(subset=[date_col], inplace=True)
    return df

@st.cache_data
def prepare_data(train, test, date_col, target_col):
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test]).sort_values(['store_nbr', 'family', date_col])
    agg_dict = {target_col: 'sum', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first'}
    combined = combined.groupby(['store_nbr', 'family', date_col]).agg(agg_dict).reset_index()
    combined = combined.astype({'store_nbr': 'int32', 'family': 'category', date_col: 'datetime64[ns]', 
                                target_col: 'float32', 'onpromotion': 'int32', 'is_train': 'int8'})
    return combined

@st.cache_data
def fill_missing(combined, target_col):
    grouped = combined.groupby(['store_nbr', 'family'])
    processed_groups = []
    for (store_nbr, family), group in grouped:
        group[target_col] = group[target_col].ffill().fillna(0).astype('float32')
        group['onpromotion'] = group['onpromotion'].fillna(0).astype('int32')
        processed_groups.append(group)
    return pd.concat(processed_groups)

@st.cache_data
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

@st.cache_data
def split_data(combined, date_col, target_col):
    train = combined[combined['is_train'] == 1]
    test = combined[combined['is_train'] == 0].drop([target_col], axis=1)
    train_set = train[train[date_col] <= TRAIN_END]
    val_set = train[(train[date_col] > TRAIN_END) & (train[date_col] <= VAL_END)]
    return train_set, val_set, test

@st.cache_data
def get_download_file(df, filename):
    buf = BytesIO()
    if not df.empty:
        df.to_csv(buf, index=False)
    else:
        buf.write(b"Data frame is empty")
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

@st.cache_data
def plot_missingness(df):
    missing_matrix = df.isna().astype(int).to_numpy()
    fig = go.Figure(data=[
        go.Heatmap(
            z=missing_matrix.T,
            x=list(range(len(df))),
            y=df.columns,
            colorscale='Viridis',
            zmin=0,
            zmax=1,
            hoverongaps=False,
            hovertemplate='Row: %{x}<br>Column: %{y}<br>Missing: %{z}<extra></extra>'
        )
    ])
    fig.update_layout(
        title="Data Missingness Overview",
        xaxis_title="Row Index",
        yaxis_title="Columns",
        yaxis_autorange='reversed'
    )
    return fig

@st.cache_data
def plot_sales_trends(df, date_col, granularity='D', family_filter=None, date_range=None):
    # Filter by family if specified
    if family_filter:
        df = df[df['family'] == family_filter]
    
    # Filter by date range if specified
    if date_range:
        start_date, end_date = date_range
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    
    # Group by the specified granularity
    sales_by_date = df.groupby(pd.Grouper(key=date_col, freq=granularity))['sales'].sum().reset_index()
    fig = px.line(sales_by_date, x=date_col, y='sales', title=f"Total Sales Trends ({granularity}ly)",
                  labels={'sales': 'Total Sales', date_col: 'Date'}, color_discrete_sequence=['blue'])
    
    # Ensure x-axis is treated as datetime
    fig.update_xaxes(type='date')
    
    # Add vertical lines for holidays
    holidays = df[df['type_y'] == 'Holiday'][date_col].unique()
    for holiday in holidays:
        # Keep holiday as a Timestamp object to match the datetime x-axis
        fig.add_vline(x=holiday, line=dict(color="red", dash="dash"), annotation_text="Holiday", annotation_position="top")
    
    fig.update_layout(xaxis=dict(tickangle=45), yaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_weekly_trends(df, date_col, family_filter=None, date_range=None):
    if family_filter:
        df = df[df['family'] == family_filter]
    if date_range:
        start_date, end_date = date_range
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    df_weekly = df.set_index(date_col)['sales'].resample('W').sum().reset_index()
    fig = px.line(df_weekly, x=date_col, y='sales', title="Weekly Sales Trends",
                  labels={'sales': 'Total Sales', date_col: 'Date'}, color_discrete_sequence=['blue'])
    fig.update_layout(xaxis=dict(tickangle=45), yaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_sales_by_family(df, color_scale='Blues', date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df = reclassify_family(df.copy())
    sales_by_family = df.groupby('family')['sales'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(sales_by_family, y='family', x='sales', orientation='h', title="Average Sales by Product Category",
                 labels={'sales': 'Average Sales', 'family': 'Product Family'}, color='sales', color_continuous_scale=color_scale)
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_sales_by_store(df, color_scale='Blues', date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    sales_by_store = df.groupby('store_nbr')['sales'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(sales_by_store, x='store_nbr', y='sales', title="Average Sales by Store Number",
                 labels={'sales': 'Average Sales', 'store_nbr': 'Store Number'}, color='sales', color_continuous_scale=color_scale)
    fig.update_layout(xaxis=dict(tickangle=45), yaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_sales_by_city_state(df, color_scale='Blues', date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df['city_state'] = df['city'] + '_' + df['state']
    sales_by_city_state = df.groupby('city_state')['sales'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(sales_by_city_state, y='city_state', x='sales', orientation='h', title="Average Sales by City-State",
                 labels={'sales': 'Average Sales', 'city_state': 'City-State'}, color='sales', color_continuous_scale=color_scale)
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_sales_by_type_locale(df, date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df['type_locale'] = df['type_y'] + '_' + df['locale']
    sales_by_type_locale = df.groupby('type_locale')['sales'].mean().reset_index()
    fig = px.pie(sales_by_type_locale, names='type_locale', values='sales', title="Sales Distribution by Type-Locale",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textinfo='percent+label', pull=[0.1 if i == sales_by_type_locale['sales'].idxmax() else 0 for i in range(len(sales_by_type_locale))])
    return fig

@st.cache_data
def plot_promotion_impact(df, color_scheme='Bold', date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    fig = px.box(df, x='onpromotion', y='sales', color='family', title="Sales Distribution by Promotion (by Family)",
                 labels={'sales': 'Sales', 'onpromotion': 'On Promotion (0 = No, 1 = Yes)'}, color_discrete_sequence=getattr(px.colors.qualitative, color_scheme))
    fig.update_layout(xaxis=dict(tickmode='linear'), yaxis_gridcolor='lightgray', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_data
def plot_sales_vs_oil(df, date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    fig = px.scatter(df, x='dcoilwtico', y='sales', trendline="ols", title="Sales vs. Oil Price with Trend",
                     labels={'sales': 'Sales', 'dcoilwtico': 'Oil Price (dcoilwtico)'}, color_discrete_sequence=['blue'])
    fig.update_layout(yaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_monthly_seasonality(df, color_scheme='Pastel', date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df['month'] = df['date'].dt.month
    sales_by_month = df.groupby(['month', 'family'])['sales'].mean().reset_index()
    fig = px.box(sales_by_month, x='month', y='sales', color='family', title="Average Monthly Sales by Family",
                 labels={'sales': 'Average Sales', 'month': 'Month'}, color_discrete_sequence=getattr(px.colors.qualitative, color_scheme))
    fig.update_layout(xaxis=dict(tickmode='linear'), yaxis_gridcolor='lightgray', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_data
def plot_dow_patterns(df, color_scheme='Pastel', date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df['dow'] = df['date'].dt.dayofweek
    sales_by_dow = df.groupby(['dow', 'family'])['sales'].mean().reset_index()
    fig = px.box(sales_by_dow, x='dow', y='sales', color='family', title="Average Sales by Day of Week (by Family)",
                 labels={'sales': 'Average Sales', 'dow': 'Day of Week (0=Mon, 6=Sun)'}, color_discrete_sequence=getattr(px.colors.qualitative, color_scheme))
    fig.update_layout(xaxis=dict(tickmode='linear'), yaxis_gridcolor='lightgray', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_data
def plot_seasonal_decomposition(df, date_col, date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    df_ts = df.set_index(date_col)['sales'].resample('M').sum()
    if len(df_ts) >= 24:
        decomp = seasonal_decompose(df_ts, model='additive', period=12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name="Trend", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name="Seasonal", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, name="Residual", line=dict(color="red")))
        fig.update_layout(title="Seasonal Decomposition of Sales", xaxis_title="Date", yaxis_title="Sales", yaxis_gridcolor='lightgray')
        return fig
    st.warning("Not enough data for seasonal decomposition (requires at least 24 months).")
    return None

@st.cache_data
def plot_autocorrelation(df, target_col, date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    n_lags = 30
    acf_vals, acf_confint = acf(df[target_col].dropna(), nlags=n_lags, alpha=0.05, fft=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF", marker_color='blue'))
    fig.add_trace(go.Scatter(x=list(range(len(acf_vals))), y=acf_confint[:, 0], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig.add_trace(go.Scatter(x=list(range(len(acf_vals))), y=acf_confint[:, 1], fill='tonexty', mode='lines', line_color='rgba(0,0,255,0.2)', name="Confidence Interval"))
    fig.update_layout(title="Autocorrelation (ACF) of Sales", xaxis_title="Lag", yaxis_title="Autocorrelation", yaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_partial_autocorrelation(df, target_col, date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    n_lags = 30
    pacf_vals, pacf_confint = pacf(df[target_col].dropna(), nlags=n_lags, alpha=0.05)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF", marker_color='blue'))
    fig.add_trace(go.Scatter(x=list(range(len(pacf_vals))), y=pacf_confint[:, 0], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig.add_trace(go.Scatter(x=list(range(len(pacf_vals))), y=pacf_confint[:, 1], fill='tonexty', mode='lines', line_color='rgba(0,0,255,0.2)', name="Confidence Interval"))
    fig.update_layout(title="Partial Autocorrelation (PACF) of Sales", xaxis_title="Lag", yaxis_title="Partial Autocorrelation", yaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_lag_plot(df, target_col, date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    lag = 1
    fig = px.scatter(df, x=target_col.shift(lag), y=target_col, trendline="ols", title=f"Lag Plot (Lag={lag})",
                     labels={f"{target_col}.shift({lag})": f"Sales (t-{lag})", target_col: "Sales (t)"}, color_discrete_sequence=['blue'])
    fig.update_layout(yaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_periodogram(df, target_col, date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    freq, psd = periodogram(df[target_col].dropna())
    fig = px.line(x=freq, y=psd, title="Periodogram of Sales",
                  labels={'x': 'Frequency', 'y': 'Power Spectral Density'}, color_discrete_sequence=['blue'])
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig.update_layout(yaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_rolling_stats(df, date_col, target - target_col, date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    rolling = df.set_index(date_col)[target_col].rolling(window=30).agg(['mean', 'std']).dropna().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling[date_col], y=rolling['mean'], name="Mean", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=rolling[date_col], y=rolling['std'], name="Std", line=dict(color="orange", dash="dot")))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig.update_layout(title="Rolling Mean and Std (30 Days)", xaxis_title="Date", yaxis_title="Sales", yaxis_gridcolor='lightgray')
    return fig

@st.cache_data
def plot_sales_heatmap(df, date_range=None):
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    pivot_data = df.pivot_table(values='sales', index='store_nbr', columns='family', aggfunc='sum', fill_value=0)
    fig = px.imshow(pivot_data, text_auto=True, aspect="auto", title="Sales Heatmap: Stores vs. Product Families",
                    labels={'x': 'Product Family', 'y': 'Store Number'}, color_continuous_scale='YlOrRd')
    fig.update_layout(xaxis=dict(side="top"), yaxis_gridcolor='lightgray')
    return fig

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
                # Read file content and hash for caching
                train_file_content = train_file.read()
                train_file.seek(0)  # Reset file pointer after reading
                train_file_hash = hash_file(train_file)
                train = load_data(train_file_content, train_file_hash, 'date', 'sales')
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
                    'train_configured': True, 'train_file': train_file, 'train_file_hash': train_file_hash
                }))

            if 'train_configured' in st.session_state and st.session_state['train_configured'] and 'train_file' in st.session_state:
                train = load_data(st.session_state['train_file'].read(), st.session_state['train_file_hash'], st.session_state['train_date_col'], st.session_state['train_target_col'])
                st.session_state['train_df'] = train
                st.dataframe(train.head(), height=100)

                # User selects plots
                plot_options = [
                    "Missingness Matrix", "Sales Trends", "Weekly Trends", "Sales by Family", "Sales by Store",
                    "Sales by City-State", "Sales by Type-Locale", "Promotion Impact", "Sales vs. Oil Price",
                    "Monthly Seasonality", "Day-of-Week Patterns", "Seasonal Decomposition", "Autocorrelation (ACF)",
                    "Partial Autocorrelation (PACF)", "Lag Plot", "Periodogram", "Rolling Statistics", "Sales Heatmap"
                ]
                selected_plots = st.multiselect("Select Plots to Generate", plot_options, default=[])

                # Configuration for each selected plot
                for plot in selected_plots:
                    with st.expander(f"Configure {plot}", expanded=True):
                        # Add date range picker for all plots
                        date_range = None
                        if plot != "Missingness Matrix":  # Missingness Matrix doesn't need date filtering
                            min_date = train[st.session_state['train_date_col']].min()
                            max_date = train[st.session_state['train_date_col']].max()
                            date_range = st.date_input(
                                "Select Date Range",
                                value=(min_date, max_date),
                                min_value=min_date,
                                max_value=max_date,
                                key=f"{plot}_date_range"
                            )
                            if len(date_range) == 2:
                                date_range = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
                            else:
                                date_range = None

                        if plot == "Missingness Matrix":
                            fig = plot_missingness(train)
                            st.plotly_chart(fig)
                        elif plot == "Sales Trends":
                            granularity = st.selectbox("Time Granularity", ['D', 'W', 'M'], index=0, key=f"{plot}_granularity")
                            family_filter = st.selectbox("Filter by Family (optional)", [None] + train['family'].unique().tolist(), index=0, key=f"{plot}_family")
                            fig = plot_sales_trends(train, st.session_state['train_date_col'], granularity, family_filter, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Weekly Trends":
                            family_filter = st.selectbox("Filter by Family (optional)", [None] + train['family'].unique().tolist(), index=0, key=f"{plot}_family")
                            fig = plot_weekly_trends(train, st.session_state['train_date_col'], family_filter, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales by Family":
                            color_scale = st.selectbox("Color Scale", ['Blues', 'Reds', 'Greens'], index=0, key=f"{plot}_color")
                            fig = plot_sales_by_family(train, color_scale, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales by Store":
                            color_scale = st.selectbox("Color Scale", ['Blues', 'Reds', 'Greens'], index=0, key=f"{plot}_color")
                            fig = plot_sales_by_store(train, color_scale, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales by City-State":
                            color_scale = st.selectbox("Color Scale", ['Blues', 'Reds', 'Greens'], index=0, key=f"{plot}_color")
                            fig = plot_sales_by_city_state(train, color_scale, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales by Type-Locale":
                            fig = plot_sales_by_type_locale(train, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Promotion Impact":
                            color_scheme = st.selectbox("Color Scheme", ['Bold', 'Pastel', 'Dark'], index=0, key=f"{plot}_color")
                            fig = plot_promotion_impact(train, color_scheme, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales vs. Oil Price":
                            fig = plot_sales_vs_oil(train, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Monthly Seasonality":
                            color_scheme = st.selectbox("Color Scheme", ['Pastel', 'Bold', 'Dark'], index=0, key=f"{plot}_color")
                            fig = plot_monthly_seasonality(train, color_scheme, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Day-of-Week Patterns":
                            color_scheme = st.selectbox("Color Scheme", ['Pastel', 'Bold', 'Dark'], index=0, key=f"{plot}_color")
                            fig = plot_dow_patterns(train, color_scheme, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Seasonal Decomposition":
                            fig = plot_seasonal_decomposition(train, st.session_state['train_date_col'], date_range)
                            if fig:
                                st.plotly_chart(fig)
                        elif plot == "Autocorrelation (ACF)":
                            fig = plot_autocorrelation(train, st.session_state['train_target_col'], date_range)
                            st.plotly_chart(fig)
                        elif plot == "Partial Autocorrelation (PACF)":
                            fig = plot_partial_autocorrelation(train, st.session_state['train_target_col'], date_range)
                            st.plotly_chart(fig)
                        elif plot == "Lag Plot":
                            fig = plot_lag_plot(train, st.session_state['train_target_col'], date_range)
                            st.plotly_chart(fig)
                        elif plot == "Periodogram":
                            fig = plot_periodogram(train, st.session_state['train_target_col'], date_range)
                            st.plotly_chart(fig)
                        elif plot == "Rolling Statistics":
                            fig = plot_rolling_stats(train, st.session_state['train_date_col'], st.session_state['train_target_col'], date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales Heatmap":
                            fig = plot_sales_heatmap(train, date_range)
                            st.plotly_chart(fig)

                csv_data, mime = get_download_file(train, "train_processed.csv")
                if csv_data and mime:
                    st.download_button("Download Processed Train Data", csv_data, "train_processed.csv", mime, key="train_download")

    with test_tab:
        test_file = st.file_uploader("Upload Test Data (.csv)", ['csv'], key="test")
        if test_file:
            if 'test_file' in st.session_state and st.session_state['test_file'] != test_file:
                for key in ['test_date_col', 'test_numeric_cols', 'test_categorical_cols', 
                           'test_outlier_method', 'test_scale', 'test_configured', 'test_df']:
                    st.session_state.pop(key, None)
            with st.form("test_config"):
                # Read file content and hash for caching
                test_file_content = test_file.read()
                test_file.seek(0)  # Reset file pointer after reading
                test_file_hash = hash_file(test_file)
                test = load_data(test_file_content, test_file_hash, 'date', None)
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
                    'test_outlier_method': outlier_method, 'test_scale': scale, 'test_configured': True, 'test_file': test_file,
                    'test_file_hash': test_file_hash
                }))

            if 'test_configured' in st.session_state and st.session_state['test_configured'] and 'test_file' in st.session_state:
                test = load_data(st.session_state['test_file'].read(), st.session_state['test_file_hash'], st.session_state['test_date_col'], None)
                st.session_state['test_df'] = test
                st.dataframe(test.head(), height=100)

                # User selects plots
                plot_options = [
                    "Missingness Matrix", "Sales Trends", "Weekly Trends", "Sales by Family", "Sales by Store",
                    "Sales by City-State", "Sales by Type-Locale", "Promotion Impact", "Sales vs. Oil Price",
                    "Monthly Seasonality", "Day-of-Week Patterns", "Seasonal Decomposition", "Autocorrelation (ACF)",
                    "Partial Autocorrelation (PACF)", "Lag Plot", "Periodogram", "Rolling Statistics", "Sales Heatmap"
                ]
                selected_plots = st.multiselect("Select Plots to Generate", plot_options, default=[])

                # Configuration for each selected plot
                for plot in selected_plots:
                    with st.expander(f"Configure {plot}", expanded=True):
                        # Add date range picker for all plots
                        date_range = None
                        if plot != "Missingness Matrix":
                            min_date = test[st.session_state['test_date_col']].min()
                            max_date = test[st.session_state['test_date_col']].max()
                            date_range = st.date_input(
                                "Select Date Range",
                                value=(min_date, max_date),
                                min_value=min_date,
                                max_value=max_date,
                                key=f"{plot}_date_range_test"
                            )
                            if len(date_range) == 2:
                                date_range = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
                            else:
                                date_range = None

                        if plot == "Missingness Matrix":
                            fig = plot_missingness(test)
                            st.plotly_chart(fig)
                        elif plot == "Sales Trends":
                            granularity = st.selectbox("Time Granularity", ['D', 'W', 'M'], index=0, key=f"{plot}_granularity_test")
                            family_filter = st.selectbox("Filter by Family (optional)", [None] + test['family'].unique().tolist(), index=0, key=f"{plot}_family_test")
                            fig = plot_sales_trends(test, st.session_state['test_date_col'], granularity, family_filter, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Weekly Trends":
                            family_filter = st.selectbox("Filter by Family (optional)", [None] + test['family'].unique().tolist(), index=0, key=f"{plot}_family_test")
                            fig = plot_weekly_trends(test, st.session_state['test_date_col'], family_filter, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales by Family":
                            color_scale = st.selectbox("Color Scale", ['Blues', 'Reds', 'Greens'], index=0, key=f"{plot}_color_test")
                            fig = plot_sales_by_family(test, color_scale, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales by Store":
                            color_scale = st.selectbox("Color Scale", ['Blues', 'Reds', 'Greens'], index=0, key=f"{plot}_color_test")
                            fig = plot_sales_by_store(test, color_scale, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales by City-State":
                            color_scale = st.selectbox("Color Scale", ['Blues', 'Reds', 'Greens'], index=0, key=f"{plot}_color_test")
                            fig = plot_sales_by_city_state(test, color_scale, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales by Type-Locale":
                            fig = plot_sales_by_type_locale(test, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Promotion Impact":
                            color_scheme = st.selectbox("Color Scheme", ['Bold', 'Pastel', 'Dark'], index=0, key=f"{plot}_color_test")
                            fig = plot_promotion_impact(test, color_scheme, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales vs. Oil Price":
                            fig = plot_sales_vs_oil(test, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Monthly Seasonality":
                            color_scheme = st.selectbox("Color Scheme", ['Pastel', 'Bold', 'Dark'], index=0, key=f"{plot}_color_test")
                            fig = plot_monthly_seasonality(test, color_scheme, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Day-of-Week Patterns":
                            color_scheme = st.selectbox("Color Scheme", ['Pastel', 'Bold', 'Dark'], index=0, key=f"{plot}_color_test")
                            fig = plot_dow_patterns(test, color_scheme, date_range)
                            st.plotly_chart(fig)
                        elif plot == "Seasonal Decomposition":
                            fig = plot_seasonal_decomposition(test, st.session_state['test_date_col'], date_range)
                            if fig:
                                st.plotly_chart(fig)
                        elif plot == "Autocorrelation (ACF)":
                            fig = plot_autocorrelation(test, st.session_state.get('train_target_col', None), date_range)
                            st.plotly_chart(fig)
                        elif plot == "Partial Autocorrelation (PACF)":
                            fig = plot_partial_autocorrelation(test, st.session_state.get('train_target_col', None), date_range)
                            st.plotly_chart(fig)
                        elif plot == "Lag Plot":
                            fig = plot_lag_plot(test, st.session_state.get('train_target_col', None), date_range)
                            st.plotly_chart(fig)
                        elif plot == "Periodogram":
                            fig = plot_periodogram(test, st.session_state.get('train_target_col', None), date_range)
                            st.plotly_chart(fig)
                        elif plot == "Rolling Statistics":
                            fig = plot_rolling_stats(test, st.session_state['test_date_col'], st.session_state.get('train_target_col', None), date_range)
                            st.plotly_chart(fig)
                        elif plot == "Sales Heatmap":
                            fig = plot_sales_heatmap(test, date_range)
                            st.plotly_chart(fig)

                csv_data, mime = get_download_file(test, "test_processed.csv")
                if csv_data and mime:
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
                if csv_data and mime:
                    st.download_button("Download Train Features", csv_data, "train_features.csv", mime, key="train_fe_download")
                csv_data, mime = get_download_file(val_set, "val_features.csv")
                if csv_data and mime:
                    st.download_button("Download Validation Features", csv_data, "val_features.csv", mime, key="val_fe_download")
                csv_data, mime = get_download_file(test, "test_features.csv")
                if csv_data and mime:
                    st.download_button("Download Test Features", csv_data, "test_features.csv", mime, key="test_fe_download")

    st.markdown("**Developed by Belal Khamis, Marwa Kotb, Mahmoud Sabry, Mohamed Samy, Hoda Magdy**")

if __name__ == "__main__":
    main()
