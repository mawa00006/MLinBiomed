import streamlit as st
from streamlit_carousel import carousel
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error
import app_helper  # Helper functions

# Define the styling for the app
st.markdown("""
    <style>
    .title {
        color: marineblue;
        font-family: 'Arial', sans-serif;
    }
    .header {
        color: marineblue;
        font-family: 'Arial', sans-serif;
    }
    .text {
        color: black;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #003366; /* Marine blue */
        color: white;
        border: none;
        font-size: 20px;
        padding: 20px;
        width: 100%;
        text-align: center;
        cursor: pointer;
        display: block;
        margin: 10px auto;
    }
    .stButton>button:hover {
        background-color: #002244; /* Darker marine blue */
    }
    .stSelectbox>div>div>div>div {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None

# Title and welcome message
st.title("Heart Stroke Analysis")
st.write("Welcome to the analysis tool. Choose an option below to get started.")

# Display the carousel
test_items = [
    dict(
        title="Slide 1",
        text="A tree in the savannah",
        img="https://www.imperial.ac.uk/news/image/mainnews2012/33304.jpg",
        link="https://www.imperial.ac.uk/news/image/mainnews2012/33304.jpg",
    ),
    dict(
        title="Slide 2",
        text="A wooden bridge in a forest in Autumn",
        img="https://th.bing.com/th/id/OIP.pyZ99psA1TuSsKRN6vBl_wHaEK?pid=ImgDet&rs=1",
        link="https://th.bing.com/th/id/OIP.pyZ99psA1TuSsKRN6vBl_wHaEK?pid=ImgDet&rs=1",
    ),
    dict(
        title="Slide 3",
        text="A distant mountain chain preceded by a sea",
        img="https://static.startuptalky.com/2021/05/ml-in-healthcare-fi-startuptalky.jpg",
        link="https://static.startuptalky.com/2021/05/ml-in-healthcare-fi-startuptalky.jpg",
    ),
]
carousel(items=test_items)

# User choice between options
st.markdown("<h2 class='header'>Choose an Analysis Type:</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("Tabular Data Analysis", key="tabular_data"):
        st.session_state.analysis_type = 'Tabular Data Analysis'

with col2:
    if st.button("Medical Imaging Pipeline", key="imaging_pipeline"):
        st.session_state.analysis_type = 'Medical Imaging Pipeline'

# Content area that changes based on the user's choice
content_area = st.empty()

# Tabular Data Analysis
if st.session_state.analysis_type == 'Tabular Data Analysis':
    with content_area.container():
        # File uploader for tabular data
        f = st.file_uploader("Please upload the dataset", type=["csv", "tsv", "xlsx", "xml", "json"])

        if f:
            # Use helper function to read the file
            df = app_helper.return_df(f)
            st.success("File Uploaded!")
            
            # EDA Section
            st.header("Explorative Data Analysis (EDA)")
            
            # Dataset preview
            st.write("Dataset preview:")
            st.dataframe(df)

            # Univariate analysis with histograms
            st.subheader("Histograms of Numeric Features")
            numeric_cols, categorical_cols = app_helper.get_column_types(df)
            app_helper.plot_numeric_histograms(df, numeric_cols)
            app_helper.plot_categorical_histograms(df, categorical_cols)

            # Correlation matrix
            st.subheader("Correlation Matrix")
            app_helper.plot_correlation_matrix(df)

            # Machine Learning Section
            st.header("Machine Learning Models")
            
            # Select target variable
            target_variable = st.selectbox("Select the target variable", df.columns)
            st.write(f"You selected **{target_variable}** as the target variable.")

            # Select models for grid search
            st.write("Select models for grid search")
            model_options = ['Logistic Regression', 'Random Forest', 'Linear Regression']
            selected_models = st.multiselect("Choose models", model_options)

            if selected_models:
                # Train/Test Split
                X, y = app_helper.get_features_and_target(df, target_variable)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                features = [col for col in df.columns if col != target_variable]
                categorical_columns = [col for col in features if df[col].dtype == 'object']
                X_train, X_test = app_helper.preprocess_data(X_train, X_test, categorical_columns)

                # Train models and show results
                model_results = app_helper.train_models(selected_models, X_train, y_train, X_test, y_test)

                for model_name, result in model_results.items():
                    st.subheader(f"Results for {model_name}")
                    st.write(result['report'])

                    # Display feature importance for Random Forest
                    if model_name == 'Random Forest':
                        st.write("Feature Importance:")
                        fig = px.bar(x=result['feature_importance'], y=X.columns, labels={'x': 'Importance', 'y': 'Feature'})
                        st.plotly_chart(fig)

                # Download options for data and models
                st.download_button(
                    label="Download Preprocessed Data",
                    data=app_helper.download_data(X, y),
                    file_name="preprocessed_data.csv"
                )

                for model_name, trained_model in model_results.items():
                    st.download_button(
                        label=f"Download {model_name} Model",
                        data=app_helper.download_model(trained_model),
                        file_name=f"{model_name}_model.pkl"
                    )

elif st.session_state.analysis_type == 'Medical Imaging Pipeline':
    with content_area.container():
        st.write("Medical Imaging Pipeline functionality has to be implemented.")