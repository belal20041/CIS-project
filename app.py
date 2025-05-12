import streamlit as st
import os
from pages import milestone3, milestone4

# Streamlit page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* General styling */
    .main {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .main-header {
        font-size: 40px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size: 28px;
        font-weight: 600;
        color: #34495E;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .description {
        font-size: 16px;
        color: #4A4A4A;
        line-height: 1.6;
        margin-bottom: 15px;
    }
    .highlight-box {
        background-color: #F8F9FA;
        padding: 15px;
        border-left: 5px solid #2E86C1;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
        padding: 20px;
        border-right: 1px solid #E0E0E0;
    }
    .sidebar-item {
        font-size: 18px;
        color: #34495E;
        padding: 10px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .sidebar-item:hover {
        background-color: #2E86C1;
        color: #FFFFFF;
        transform: scale(1.05);
    }
    /* Buttons */
    .stButton>button {
        background-color: #2E86C1;
        color: #FFFFFF;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1B5E91;
        transform: scale(1.05);
    }
    /* Radio buttons */
    .stRadio > div {
        background-color: #F8F9FA;
        padding: 10px;
        border-radius: 5px;
    }
    .stRadio > div > label {
        font-size: 16px;
        color: #34495E;
        padding: 8px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stRadio > div > label:hover {
        background-color: #E8ECEF;
    }
    /* Footer */
    .footer {
        text-align: center;
        font-size: 14px;
        color: #7F8C8D;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #E0E0E0;
    }
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 32px;
        }
        .sub-header {
            font-size: 24px;
        }
        .description {
            font-size: 14px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation with radio buttons
st.sidebar.title("Sales Forecasting and Optimization")
st.sidebar.markdown("<div class='description'>Navigate through the project milestones:</div>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Select a Milestone",
    [
        "Home",
        "Milestone 1: Data Collection, Exploration, Preprocessing",
        "Milestone 2: Data Analysis and Visualization",
        "Milestone 3: Forecasting Model Development and Optimization",
        "Milestone 4: MLOps, Deployment, and Monitoring",
        "Milestone 5: Final Documentation and Presentation"
    ],
    index=0
)

# Main container
with st.container():
    # Logo or header
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=200, use_column_width=False, caption="Sales Forecasting Dashboard")
    else:
        st.markdown("<div class='main-header'>Sales Forecasting Dashboard</div>", unsafe_allow_html=True)

    # Page content
    if page == "Home":
        # Introduction
        st.markdown("""
            <div class='highlight-box'>
                <div class='description'>
                    Welcome to the <b>Sales Forecasting Dashboard</b>, a comprehensive tool for predicting retail sales using advanced machine learning and time-series models. Built for the CIS project, this app supports the full lifecycle of sales forecasting, from data preprocessing to model deployment and monitoring, powered by Streamlit and MLflow.
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Project overview
        st.markdown("<div class='sub-header'>Project Overview</div>", unsafe_allow_html=True)
        st.markdown("""
            <div class='description'>
                This dashboard enables you to forecast sales for retail stores using models like XGBoost, ARIMA, and Prophet. It includes data preprocessing, feature engineering, model training, real-time/batch predictions, and performance monitoring with MLOps practices. Explore the milestones via the sidebar to access each phase of the project.
            </div>
        """, unsafe_allow_html=True)

        # Available milestones
        st.markdown("<div class='sub-header'>Explore the Milestones</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class='highlight-box'>
                    <b>Milestone 3: Forecasting Model Development</b><br>
                    <div class='description'>
                        Upload your dataset, specify columns, and train forecasting models. Visualize time-series analysis and evaluate performance with metrics like RMSLE and RMSE.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Milestone 3"):
                st.session_state.page = "Milestone 3: Forecasting Model Development and Optimization"
        with col2:
            st.markdown("""
                <div class='highlight-box'>
                    <b>Milestone 4: MLOps, Deployment, Monitoring</b><br>
                    <div class='description'>
                        Deploy models for real-time or batch predictions, track experiments with MLflow, and monitor performance for drift detection and accuracy alerts.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Milestone 4"):
                st.session_state.page = "Milestone 4: MLOps, Deployment, and Monitoring"

        # Instructions (collapsible)
        with st.expander("How to Use This Dashboard", expanded=False):
            st.markdown("""
                <div class='description'>
                    <ul>
                        <li><b>Step 1</b>: Use the sidebar to select a milestone or stay on the Home page for an overview.</li>
                        <li><b>Step 2</b>: For Milestones 3 and 4, upload your dataset (CSV or Parquet) and specify columns (e.g., date, target, numeric, categorical).</li>
                        <li><b>Step 3</b>: Follow the prompts to train models, generate predictions, or monitor performance.</li>
                        <li><b>Step 4</b>: Ensure dependencies are installed using <code>D:\\DEPI\\CIS project\\deployment\\requirements.txt</code>.</li>
                        <li><b>Step 5</b>: Save outputs (e.g., submissions, MLflow artifacts) to <code>D:\\DEPI\\CIS project</code>.</li>
                    </ul>
                    <b>Note</b>: Place data files (e.g., Train.csv, Test.csv, Submission.csv) in <code>D:\\DEPI\\CIS project</code>. Milestones 1, 2, and 5 are under development.
                </div>
            """, unsafe_allow_html=True)

    elif page.startswith("Milestone 1"):
        st.markdown("<div class='sub-header'>Milestone 1: Data Collection, Exploration, Preprocessing</div>", unsafe_allow_html=True)
        st.warning("This milestone is under development. Please check back later or proceed to Milestones 3 or 4.")

    elif page.startswith("Milestone 2"):
        st.markdown("<div class='sub-header'>Milestone 2: Data Analysis and Visualization</div>", unsafe_allow_html=True)
        st.warning("This milestone is under development. Please check back later or proceed to Milestones 3 or 4.")

    elif page.startswith("Milestone 3"):
        try:
            milestone3.main()
        except Exception as e:
            st.error(f"Error loading Milestone 3: {str(e)}. Ensure `milestone3.py` is in `D:\\DEPI\\CIS project\\pages` and has a `main()` function.")

    elif page.startswith("Milestone 4"):
        try:
            milestone4.main()
        except Exception as e:
            st.error(f"Error loading Milestone 4: {str(e)}. Ensure `milestone4.py` is in `D:\\DEPI\\CIS project\\pages` and has a `main()` function.")

    elif page.startswith("Milestone 5"):
        st.markdown("<div class='sub-header'>Milestone 5: Final Documentation and Presentation</div>", unsafe_allow_html=True)
        st.warning("This milestone is under development. Please check back later or proceed to Milestones 3 or 4.")

# Check for pages directory
pages_dir = os.path.join(os.path.dirname(__file__), "pages")
if not os.path.exists(pages_dir):
    st.error(f"Pages directory not found at {pages_dir}. Please create `D:\\DEPI\\CIS project\\pages` and add `milestone3.py` and `milestone4.py`.")

# Footer
st.markdown("""
    <div class='footer'>
        Developed for CIS Project | Powered by Streamlit | © 2025
    </div>
""", unsafe_allow_html=True)