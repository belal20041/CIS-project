import streamlit as st

# Set page configuration for the entire app
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="auto"
)

# Main app title and introduction
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>Sales Forecasting Dashboard</h1>
    <p style='text-align: center; font-size: 16px; margin-top: 10px;'>
        This dashboard provides an end-to-end workflow for sales forecasting, 
        from data exploration to model deployment. Navigate to the milestones 
        using the sidebar to process data, engineer features, train models, and 
        monitor performance.
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Milestones")
st.sidebar.markdown(
    """
    Select a milestone to proceed:
    - **Milestone 1**: Data Collection, Exploration, Preprocessing
    - **Milestone 2**: Feature Engineering
    - **Milestone 3**: Model Training & Forecasting
    - **Milestone 4**: MLOps, Deployment, and Monitoring
    """
)

# Sidebar instructions
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Instructions**:
    1. Upload CSV or Parquet files as prompted.
    2. Select relevant columns for analysis.
    3. Follow the steps in each milestone.
    Ensure the project directory (`D:\\DEPI\\CIS project`) is accessible.
    """
)

# Main content
st.markdown("### Overview")
st.write(
    """
    This application is designed to support a sales forecasting project with the following stages:
    - **Data Preprocessing**: Load, clean, and explore datasets.
    - **Feature Engineering**: Create features for predictive modeling.
    - **Model Training**: Train forecasting models like XGBoost, ARIMA, and Prophet.
    - **MLOps & Deployment**: Deploy models, generate predictions, and monitor performance.

    Use the sidebar to select a milestone and begin. Outputs are saved to the `data`, `models`, and `mlruns` directories.
    """
)

# Footer
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; font-size: 12px; color: #666;'>
        Powered by Streamlit | Project Path: D:\\DEPI\\CIS project | Created for Sales Forecasting
    </p>
    """,
    unsafe_allow_html=True
)
