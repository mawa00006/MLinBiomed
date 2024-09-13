import streamlit as st
from streamlit_carousel import carousel
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
import pandas as pd
import app_helper  # Helper functions
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error
import app_helper  # Helper functions
import pickle as pkl

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
st.title("Medical Data Analysis Tool")
st.markdown("<p style='font-size: 14px; color: black;'>by Elisa Maske and Mattes Warning</p>", unsafe_allow_html=True)
st.markdown("<h3>Welcome to the medical data analysis tool. Choose an option below to get started.</h3>", unsafe_allow_html=True)

# Display the carousel
test_items = [
    dict(
        title="",
        text="",
        img="https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Parasagittal_MRI_of_human_head_in_patient_with_benign_familial_macrocephaly_prior_to_brain_injury_%28ANIMATED%29.gif/640px-Parasagittal_MRI_of_human_head_in_patient_with_benign_familial_macrocephaly_prior_to_brain_injury_%28ANIMATED%29.gif",
        link="https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Parasagittal_MRI_of_human_head_in_patient_with_benign_familial_macrocephaly_prior_to_brain_injury_%28ANIMATED%29.gif/640px-Parasagittal_MRI_of_human_head_in_patient_with_benign_familial_macrocephaly_prior_to_brain_injury_%28ANIMATED%29.gif",
    ),
    dict(
        title="",
        text="",
        img="https://upload.wikimedia.org/wikipedia/commons/6/61/Apikal4D.gif?20080313194245",
        link="https://upload.wikimedia.org/wikipedia/commons/6/61/Apikal4D.gif?20080313194245",
    ),
    dict(
        title="",
        text="",
        img="https://images.unsplash.com/photo-1584555613497-9ecf9dd06f68?q=80&w=2050&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        link="https://images.unsplash.com/photo-1584555613497-9ecf9dd06f68?q=80&w=2050&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    ),
]

carousel(items=test_items)

# User choice between options
st.markdown("<h2 class='header'>Choose an Analysis Type:</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("Model Training and Evaluation", key="tabular_data"):
        st.session_state.analysis_type = 'Model Training and Evaluation'

with col2:
    if st.button("Prediction and Analysis", key="imaging_pipeline"):
        st.session_state.analysis_type = 'Prediction and Analysis'

# Content area that changes based on the user's choice
content_area = st.empty()

# Tabular Data Analysis
if st.session_state.analysis_type == 'Model Training and Evaluation':
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
            target_cols = [col for col in df.columns if col != 'id']
            target_variable = st.selectbox("Select the target variable", target_cols)
            st.write(f"You selected **{target_variable}** as the target variable.")

            # Select models for grid search
            st.write("Select models for grid search")
            model_options = ['Logistic Regression', 'Random Forest', 'Linear Regression']
            selected_models = st.multiselect("Choose models", model_options)

            if selected_models:
                param_grids = app_helper.create_param_grids(selected_models)

            if selected_models and param_grids!={}:
                # Train/Test Split
                X, y = app_helper.get_features_and_target(df, target_variable)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
                X_train, X_test, encoder = app_helper.preprocess_data(X_train, X_test, categorical_columns)

                # Save the encoder for later use
                with open('encoder.pkl', 'wb') as f:
                    pkl.dump(encoder, f)

                # Train models and show results
                model_results = app_helper.train_models(selected_models, X_train, y_train, X_test, y_test, param_grids)

                for model_name, result in model_results.items():
                    st.subheader(f"Results for {model_name}")
                    st.text("──"+result['report'])


                    # Display feature importance for Random Forest
                    if model_name == 'Random Forest':
                          feature_importances = result['feature_importance']
                          feature_importance_df = pd.DataFrame({
                                'Feature': X_train.columns,
                                'Importance': feature_importances
                                })
                          feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
                          fig = px.bar(feature_importance_df, x='Feature', y='Importance', 
                                       labels={'x': 'Feature', 'y': 'Importance'},
                                       title='Feature Importances',
                                       color_discrete_sequence=['#00008B'])  # Dark blue hex code
                          st.plotly_chart(fig)

                # Download options for data and models
                st.download_button(
                    label="Download Preprocessed Data",
                    data=app_helper.download_data(X, y),
                    file_name="preprocessed_data.csv"
                )

                for model_name, model_info in model_results.items():
                     trained_model = model_info['model']  # Extract the actual trained model
                     st.download_button(label=f"Download {model_name} Model",
                                        data=app_helper.download_model(trained_model),
                                        file_name=f"{model_name}_model.pkl"
                                        )

                st.write("Model training and evaluation completed!")


elif st.session_state.analysis_type == 'Prediction and Analysis':
    with content_area.container():
        st.write("Upload the trained model and a new dataset for predictions.")
        
        # Upload the trained model
        uploaded_model = st.file_uploader("Upload Trained Model (pkl file)", type="pkl")

        if uploaded_model:
            # Load the uploaded model
            model = app_helper.load_model(uploaded_model)
            st.success("Model uploaded successfully!")
            
            # Upload the new dataset
            f_new_data = st.file_uploader("Upload New Dataset for Prediction", type=["csv", "tsv", "xlsx", "xml", "json"])
            

            if f_new_data:
                # Use helper function to read the file
                df_new_data = app_helper.return_df(f_new_data)
                st.success("New dataset uploaded!")

                # Store 'id' and 'former_target' for later
                id_column = 'id' if 'id' in df_new_data.columns else None
                former_target = st.selectbox("Select the former target variable that was also used during the training process", df_new_data.columns, index=11)
                st.write(f"You selected **{former_target}** as the target variable.")
    
                # Keep 'id' and drop 'former_target' from the dataset before predictions
                df_new_data_for_prediction = df_new_data.drop(columns=[former_target, 'id'], errors='ignore')

                # Preprocess new data (assuming similar structure)
                with open('encoder.pkl', 'rb') as f:
                    encoder = pkl.load(f)
                categorical_columns = [col for col in df_new_data_for_prediction.columns if df_new_data_for_prediction[col].dtype == 'object']
                X_new_data = app_helper.preprocess_new_data(df_new_data_for_prediction, categorical_columns, encoder)

                # Make predictions
                st.write("Making predictions...")
                predictions = model.predict(X_new_data)

                # Reattach the 'id' column and display the predictions in a more informative way
                if id_column:
                    df_results = df_new_data[[id_column]].copy()
                    df_results['Prediction'] = predictions
                else:
                    df_results = pd.DataFrame({'Prediction': predictions})

                # Display predictions with IDs
                st.subheader("Predictions with Patient IDs")
                st.dataframe(df_results)

                # Display summary statistics
                st.subheader("Summary Statistics")
                stroke_percentage = (df_results['Prediction'].mean()) * 100
                st.write(f"{stroke_percentage:.2f}% of the patients in the dataset is predicted to have a stroke.")
                st.write(f"{100 - stroke_percentage:.2f}% of the patients in the dataset is predicted not to have a stroke.")

                # Option to download predictions
                st.download_button(
                    label="Download Predictions with IDs",
                    data=app_helper.download_predictions(df_results),
                    file_name="predictions_with_ids.csv"
                )

                st.write("Predictions completed!")