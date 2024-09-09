import streamlit as st
from streamlit_carousel import carousel
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px

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

def return_df(file):
    name = file.name
    extension = name.split(".")[-1]
    if extension == "csv":
        df = pd.read_csv(name)
    elif extension == "tsv":
        df = pd.read_csv(name, sep="\t")
    elif extension == "xlsx":
        df = pd.read_excel(name)
    elif extension == "xml":
        df = pd.read_xml(name)
    elif extension == "json":
        df = pd.read_json(name)
    return df

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

if st.session_state.analysis_type == 'Tabular Data Analysis':
    with content_area.container():
        # File uploader for tabular data
        f = st.file_uploader("Please upload the dataset", type=["csv", "tsv", "xlsx", "xml", "json"])

        if f:
            df = return_df(f)
            st.success("File Uploaded!")
            st.write("Dataset preview:")
            st.dataframe(df)

            # Task type selection (Classification/ Regression)
            task_type = st.radio("Select task type", ('Classification', 'Regression'))

            # Select target variable
            target_variable = st.selectbox("Select the target variable", df.columns)

            # Model selection
            st.write("Select models for grid search")
            models = st.multiselect("Choose models", ['Logistic Regression', 'Random Forest', 'SVM', 'KNN', 'Linear Regression'])

            # Create tabs
            tab1, tab2 = st.tabs(["EDA", "3D Visualization and Prediction"])

            with tab1:
                # EDA Tab
                pr = ProfileReport(df)
                st_profile_report(pr)

            with tab2:
                # 3D Visualization and Prediction Tab
                col_fea1, col_fea2, col_fea3, col_target = st.columns(4)
                with col_fea1:
                    fea1 = st.selectbox("Please select the first feature", df.columns)
                with col_fea2:
                    fea2 = st.selectbox("Please select the second feature", df.columns)
                with col_fea3:
                    fea3 = st.selectbox("Please select the third feature", df.columns)
                with col_target:
                    target = st.selectbox("Please select the target", df.columns)
                
                # 3D Plot
                fig_3d = px.scatter_3d(df, x=fea1, y=fea2, z=fea3, color=target)
                st.plotly_chart(fig_3d)

                # Personalized Input and Prediction
                st.subheader("Personalized Input")
                smoking_status = st.radio("Do you smoke?", ('Yes', 'No'))
                st.write(f"Selected smoking status: {smoking_status}")
                
                # Placeholder for prediction logic
                # Replace this with your actual model and prediction logic
                st.write("Prediction results will be displayed here.")

elif st.session_state.analysis_type == 'Medical Imaging Pipeline':
    with content_area.container():
        st.write("Medical Imaging Pipeline functionality has to be implemented.")
