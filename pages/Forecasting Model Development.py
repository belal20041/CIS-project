import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title="Sales Forecasting", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Training", "Testing", "Future Forecast"])

@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv", parse_dates=['Date'])
    df.sort_values("Date", inplace=True)
    return df

data = load_data()

# Common preprocessing
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
features = ['Day', 'Month', 'Year']
X = data[features]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tabs
if tab == "Training":
    st.title("Model Training")

    st.write("### Data Preview")
    st.dataframe(data.head())

    st.write("### Feature Importance")
    importance = pd.Series(model.feature_importances_, index=features)
    fig, ax = plt.subplots()
    importance.plot(kind='barh', ax=ax)
    st.pyplot(fig)

    st.write("### Training Metrics")
    y_pred_train = model.predict(X_train)
    st.write(f"MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
    st.write(f"MSE: {mean_squared_error(y_train, y_pred_train):.2f}")

elif tab == "Testing":
    st.title("Model Testing")

    st.write("### Test Metrics")
    y_pred_test = model.predict(X_test)
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred_test):.2f}")

    st.write("### Predictions vs Actual")
    result_df = pd.DataFrame({
        'Date': data.loc[y_test.index, 'Date'],
        'Actual': y_test.values,
        'Predicted': y_pred_test
    })
    result_df.set_index('Date', inplace=True)
    st.line_chart(result_df)

elif tab == "Future Forecast":
    st.title("Future Forecast")

    st.write("### Forecast Settings")
    start_date = data['Date'].max() + timedelta(days=1)
    forecast_days = st.slider("Select number of days to forecast", 1, 60, 30)

    future_dates = pd.date_range(start=start_date, periods=forecast_days)
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Day': future_dates.day,
        'Month': future_dates.month,
        'Year': future_dates.year
    })

    future_pred = model.predict(future_df[['Day', 'Month', 'Year']])
    forecast_result = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Sales': future_pred
    })
    forecast_result.set_index('Date', inplace=True)

    st.line_chart(forecast_result)
    st.write("### Forecast Data")
    st.dataframe(forecast_result.reset_index())
