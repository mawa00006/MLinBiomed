import pandas as pd
import pickle
import io
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt

def return_df(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        return pd.read_csv(file)
    elif file_extension == 'tsv':
        return pd.read_csv(file, sep='\t')
    elif file_extension == 'xlsx':
        return pd.read_excel(file)
    elif file_extension == 'xml':
        return pd.read_xml(file)
    elif file_extension == 'json':
        return pd.read_json(file)
    else:
        raise ValueError("Unsupported file format")

def get_column_types(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols

def plot_numeric_histograms(df, numeric_cols):
    if numeric_cols:
        df[numeric_cols].hist(bins=30, figsize=(15, 10), color='skyblue', edgecolor='black', grid=False)
        plt.tight_layout()
        st.pyplot()

def plot_categorical_histograms(df, categorical_cols):
    for col in categorical_cols:
        df[col].value_counts().plot(kind='bar', figsize=(10, 5))
        plt.title(f'Bar plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        st.pyplot()

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot()

def preprocess_data(X_train, X_test, categorical_columns):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])
    
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
    X_train = X_train.drop(columns=categorical_columns).reset_index(drop=True)
    X_test = X_test.drop(columns=categorical_columns).reset_index(drop=True)
    
    X_train_encoded_df = pd.concat([X_train, X_train_encoded_df], axis=1)
    X_test_encoded_df = pd.concat([X_test, X_test_encoded_df], axis=1)
    
    return X_train_encoded_df, X_test_encoded_df


def get_features_and_target(df, target_variable):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    return X, y

def preprocess_data(X_train, X_test, categorical_columns):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])
    
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
    X_train = X_train.drop(columns=categorical_columns).reset_index(drop=True)
    X_test = X_test.drop(columns=categorical_columns).reset_index(drop=True)
    
    X_train_encoded_df = pd.concat([X_train, X_train_encoded_df], axis=1)
    X_test_encoded_df = pd.concat([X_test, X_test_encoded_df], axis=1)
    
    return X_train_encoded_df, X_test_encoded_df

def train_models(selected_models, X_train, y_train, X_test, y_test):
    # Import required modules
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, mean_squared_error
    from sklearn.model_selection import GridSearchCV

    # Initialize dictionaries to store parameter grids and models
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'class_weight': ['balanced', None]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'class_weight': ['balanced', None]
        }
    }
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Linear Regression': LinearRegression()
    }

    best_estimators = {}

    for model_name in selected_models:
        model = models.get(model_name)
        param_grid = param_grids.get(model_name, {})
        
        scoring = 'f1_macro' if model_name != 'Linear Regression' else 'neg_mean_squared_error'
                
        # Train the model with GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_estimators[model_name] = grid_search.best_estimator_
        

    results = {}
    for model_name, model in best_estimators.items():
        try:
            y_pred = model.predict(X_test)
            if model_name == 'Linear Regression':
                results[model_name] = {
                    'report': f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}',
                    'model': model
                }
            else:
                results[model_name] = {
                    'report': classification_report(y_test, y_pred),
                    'model': model,
                    'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
                }
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")

    return results




def download_data(X, y):
    df = pd.concat([X, y], axis=1)
    csv = df.to_csv(index=False)
    return StringIO(csv).getvalue()

def download_model(model):
    return pickle.dumps(model)


"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocess_data(X_train, X_test, categorical_columns):
    # Erstelle einen OneHotEncoder für die kategorialen Spalten
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit auf die Trainingsdaten
    X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Transformiere die Testdaten
    X_test_encoded = encoder.transform(X_test[categorical_columns])
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Drop die ursprünglichen kategorialen Spalten und füge die kodierten Spalten hinzu
    X_train = X_train.drop(columns=categorical_columns).reset_index(drop=True)
    X_test = X_test.drop(columns=categorical_columns).reset_index(drop=True)
    
    X_train_encoded_df = pd.concat([X_train, X_train_encoded_df], axis=1)
    X_test_encoded_df = pd.concat([X_test, X_test_encoded_df], axis=1)
    
    return X_train_encoded_df, X_test_encoded_df


# Function to read different file formats
def return_df(file):
    name = file.name
    extension = name.split(".")[-1]
    if extension == "csv":
        df = pd.read_csv(file)
    elif extension == "tsv":
        df = pd.read_csv(file, sep="\t")
    elif extension == "xlsx":
        df = pd.read_excel(file)
    elif extension == "xml":
        df = pd.read_xml(file)
    elif extension == "json":
        df = pd.read_json(file)
    return df

# Function to get numeric and categorical column types
def get_column_types(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols

# Function to plot numeric histograms
def plot_numeric_histograms(df, numeric_cols):
    if numeric_cols:
        df[numeric_cols].hist(bins=30, figsize=(15, 10), color='skyblue', edgecolor='black', grid=False)
        plt.tight_layout()
        st.pyplot()

# Function to plot categorical histograms
def plot_categorical_histograms(df, categorical_cols):
    if categorical_cols:
        plt.figure(figsize=(15, 12))
        for i, col in enumerate(categorical_cols, 1):
            plt.subplot(3, 2, i)
            df[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {col}')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot()

# Function to plot the correlation matrix
def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot()

# Function to separate features and target
def get_features_and_target(df, target_variable):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    return X, y

# Function to train models and return results
def train_models(selected_models, X_train, y_train, X_test, y_test):
    results = {}
    for model in selected_models:
        if model == 'Logistic Regression':
            clf = LogisticRegression()
        elif model == 'Random Forest':
            clf = RandomForestClassifier() if y_train.nunique() > 2 else RandomForestRegressor()
        elif model == 'Linear Regression':
            clf = LinearRegression()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if model == 'Random Forest':
            results[model] = {
                'report': classification_report(y_test, y_pred) if y_train.nunique() > 2 else mean_squared_error(y_test, y_pred),
                'feature_importance': clf.feature_importances_ if hasattr(clf, 'feature_importances_') else None
            }
        else:
            results[model] = {'report': classification_report(y_test, y_pred) if y_train.nunique() > 2 else mean_squared_error(y_test, y_pred)}

    return results

# Function to prepare the preprocessed data for download
def download_data(X, y):
    output = BytesIO()
    pd.concat([X, y], axis=1).to_csv(output, index=False)
    return output.getvalue()

# Function to prepare a trained model for download
def download_model(model):
    output = BytesIO()
    pickle.dump(model, output)
    return output.getvalue()
"""