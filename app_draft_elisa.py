import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px


def return_df(file):
    name = file.name
    extension = name.split(".")[-1]
    if extension == "csv":
        df = pd.read_csv(name)
    elif extension == "tsv":
        df = pd.read_csv(name, sep ="\t")
    elif extension == "xlsx":
        df = pd.read_excel(name)
    elif extension == "xml":
        df = pd.read_xml(name)
    elif extension == "json":
        df = pd.read_json(name)
    return df

st.title("EDA for Heart Stroke Analysis")
st.write("This is your first Streamlit app.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3>Automated Analysis</h3>", unsafe_allow_html = True)
    st.markdown("<p>Automated Analysis</p>", unsafe_allow_html = True)
with col2:
    st.markdown("<h3>Automated Analysis</h3>", unsafe_allow_html = True)
    st.markdown("<p>Automated Analysis</p>", unsafe_allow_html = True)
with col3:
    st.markdown("<h3>Automated Analysis</h3>", unsafe_allow_html = True)
    st.markdown("<p>Automated Analysis</p>", unsafe_allow_html = True)



st.video("https://www.youtube.com/watch?v=3_PYnWVoUzM")



from streamlit_carousel import carousel

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

f = st.file_uploader("Please upload the dataset", type=["csv", "tsv", "xlsx", "xml", "json"])
print(f)

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

    tab1, tab2 = st.tabs(["EDA", "3D Visualization"])

    with tab1:
        pr = ProfileReport(df)
        st_profile_report(pr)
    with tab2:
        col_fea1,col_fea2,col_fea3,col_target = st.columns(4)
        with col_fea1:
            fea1 = st.selectbox("Please select the first feature",df.columns)
        with col_fea2:
            fea2 = st.selectbox("Please select the second feature",df.columns)
        with col_fea3:
            fea3 = st.selectbox("Please select the third feature",df.columns)
        with col_target:
            target = st.selectbox("Please select the target",df.columns)
        

        fig_3d = px.scatter_3d(df, x=fea1,y=fea2,z=fea3,color=target)

        st.plotly_chart(fig_3d)