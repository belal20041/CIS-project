import streamlit as st
from pages import milestone1, milestone2, milestone3, milestone4, milestone5
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.sidebar.title("Sales Forecasting and Optimization")
page = st.sidebar.radio("Navigate to", [
    "Milestone 1: Data Collection, Exploration, Preprocessing",
    "Milestone 2: Data Analysis and Visualization",
    "Milestone 3: Forecasting Model Development and Optimization",
    "Milestone 4: MLOps, Deployment, and Monitoring",
    "Milestone 5: Final Documentation and Presentation"
])
if page.startswith("Milestone 1"):
    milestone1.main()
elif page.startswith("Milestone 2"):
    milestone2.main()
elif page.startswith("Milestone 3"):
    milestone3.main()
elif page.startswith("Milestone 4"):
    milestone4.main()
elif page.startswith("Milestone 5"):
    milestone5.main()
