import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from pmdarima import auto_arima
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import psutil
import warnings
warnings.filterwarnings("ignore")

# Streamlit app title
st.title("Sales Forecasting Dashboard")

# Initialize session state to store results
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'train_set' not in st.session_state:
    st.session_state.train_set = None
if 'val_set' not in st.session_state:
    st.session_state.val_set = None
if 'test' not in st.session_state:
    st.session_state.test = None
if 'sub' not in st.session_state:
    st.session_state.sub = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None

# Tabs
training_tab, prediction_tab = st.tabs(["Training", "Prediction"])

# Constants
TRAIN_END = '2017-07-15'
VAL_END = '2017-08-15'

# Cache data loading and processing
@st.cache_data
def load_and_process_data(train_file, test_file, sub_file):
    try:
        # Load data
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        sub = pd.read_csv(sub_file)
        
        # Data preprocessing (from original Cells 3–6)
        train['date'] = pd.to_datetime(train['date'])
        test['date'] = pd.to_datetime(test['date'], format='%d-%m-%Y')
        train[['store_nbr', 'onpromotion']] = train[['store_nbr', 'onpromotion']].astype('int32')
        test[['store_nbr', 'onpromotion']] = test[['store_nbr', 'onpromotion']].astype('int32')
        train['sales'] = train['sales'].astype('float32')
        train.dropna(subset=['date'], inplace=True)
        test.dropna(subset=['date'], inplace=True)
        
        # Prepare data
        train['is_train'] = 1
        test['is_train'] = 0
        combined = pd.concat([train, test]).sort_values(['store_nbr', 'family', 'date'])
        agg_dict = {'sales': 'mean', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first'}
        combined = combined.groupby(['store_nbr', 'family', 'date']).agg(agg_dict).reset_index()
        combined = combined.astype({'store_nbr': 'int32', 'family': 'category', 'date': 'datetime64[ns]', 
                                   'sales': 'float32', 'onpromotion': 'int32', 'is_train': 'int8'})
        
        # Handle missing values
        grouped = combined.groupby(['store_nbr', 'family'])
        processed_groups = []
        for (store_nbr, family), group in grouped:
            group['sales'] = group['sales'].ffill().fillna(0).astype('float32')
            group['onpromotion'] = group['onpromotion'].fillna(0).astype('int32')
            processed_groups.append(group)
        combined = pd.concat(processed_groups)
        
        # Add features
        combined['day'] = combined['date'].dt.day.astype('int8')
        combined['dow'] = combined['date'].dt.dayofweek.astype('int8')
        combined['month'] = combined['date'].dt.month.astype('int8')
        combined['year'] = combined['date'].dt.year.astype('int16')
        combined['sin_month'] = np.sin(2 * np.pi * combined['month'] / 12).astype('float32')
        lags = [7, 14]
        for lag in lags:
            combined[f'lag_{lag}'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(lag).astype('float32')
        combined['roll_mean_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7, min_periods=1).mean().astype('float32')
        combined['store_nbr_encoded'] = LabelEncoder().fit_transform(combined['store_nbr']).astype('int8')
        combined['family_encoded'] = LabelEncoder().fit_transform(combined['family']).astype('int8')
        feature_cols = ['onpromotion', 'day', 'dow', 'month', 'year', 'sin_month', 'store_nbr_encoded', 
                        'family_encoded', 'lag_7', 'lag_14', 'roll_mean_7']
        combined[feature_cols] = StandardScaler().fit_transform(combined[feature_cols].fillna(0)).astype('float32')
        
        # Split data
        train = combined[combined['is_train'] == 1]
        test = combined[combined['is_train'] == 0].drop(['sales'], axis=1)
        train_set = train[train['date'] <= TRAIN_END]
        val_set = train[(train['date'] > TRAIN_END) & (train['date'] <= VAL_END)]
        
        return train_set, val_set, test, sub, feature_cols
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None, None, None, None, None

# Training Tab
with training_tab:
    st.header("Train Forecasting Models")
    
    # File uploaders
    st.subheader("Upload Data")
    train_file = st.file_uploader("Upload Train CSV", type="csv", key="uploader_train")
    test_file = st.file_uploader("Upload Test CSV", type="csv", key="uploader_test")
    sub_file = st.file_uploader("Upload Submission CSV", type="csv#pragma once

# include <string>
#include <vector>
#include <map>
#include <ctime>
#include <memory>

class TimeSeriesModel {
public:
    virtual ~TimeSeriesModel() = default;
    virtual void fit(const std::vector<double>& data) = 0;
    virtual std::vector<double> predict(int steps) const = 0;
    virtual std::string getName() const = 0;
};

class NaiveForecast : public TimeSeriesModel {
public:
    void fit(const std::vector<double>& data) override;
    std::vector<double> predict(int steps) const override;
    std::string getName() const override { return "Naive"; }

private:
    double last_value = 0.0;
};

class SeasonalNaiveForecast : public TimeSeriesModel {
public:
    SeasonalNaiveForecast(int season_length = 7) : season_length(season_length) {}
    void fit(const std::vector<double>& data) override;
    std::vector<double> predict(int steps) const override;
    std::string getName() const override { return "Seasonal Naive"; }

private:
    int season_length;
    std::vector<double> last_season;
};

class ExponentialSmoothing : public TimeSeriesModel {
public:
    ExponentialSmoothing(double alpha = 0.3) : alpha(alpha) {}
    void fit(const std::vector<double>& data) override;
    std::vector<double> predict(int steps) const override;
    std::string getName() const override { return "Exponential Smoothing"; }

private:
    double alpha;
    double level = 0.0;
};

class HoltsLinearTrend : public TimeSeriesModel {
public:
    HoltsLinearTrend(double alpha = 0.3, double beta = 0.1) : alpha(alpha), beta(beta) {}
    void fit(const std::vector<double>& data) override;
    std::vector<double> predict(int steps) const override;
    std::string getName() const override { return "Holt's Linear Trend"; }

private:
    double alpha, beta;
    double level = 0.0, trend = 0.0;
};

class MovingAverageForecast : public TimeSeriesModel {
public:
    MovingAverageForecast(int window = 7) : window(window) {}
    void fit(const std::vector<double>& data) override;
    std::vector<double> predict(int steps) const override;
    std::string getName() const override { return "Moving Average"; }

private:
    int window;
    double last_average = 0.0;
};

class LinearRegressionTS : public TimeSeriesModel {
public:
    void fit(const std::vector<double>& data) override;
    std::vector<double> predict(int steps) const override;
    std::string getName() const override { return "Linear Regression"; }

private:
    std::vector<double> coefficients;
    double intercept = 0.0;
    std::vector<double> create_features(int index, int lag = 7) const;
};

class TimeSeriesForecaster {
public:
    TimeSeriesForecaster();
    void addModel(std::unique_ptr<TimeSeriesModel> model);
    void fitAll(const std::map<std::pair<int, std::string>, std::vector<double>>& data);
    std::map<std::string, std::vector<double>> predictAll(const std::pair<int, std::string>& key, int steps) const;
    std::vector<std::string> getModelNames() const;

private:
    std::vector<std::unique_ptr<TimeSeriesModel>> models;
};

#endif // TIME_SERIES_MODELS_H
