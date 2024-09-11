import pandas as pd
import pickle
from io import StringIO
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
        fig, ax = plt.subplots(figsize=(15, 10))
        df[numeric_cols].hist(bins=30, color='skyblue', edgecolor='black', grid=False, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)


def plot_categorical_histograms(df, categorical_cols):
    if categorical_cols:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
        axes = axes.flatten()
        for i, col in enumerate(categorical_cols):
            ax = axes[i]
            df[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

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
                # Ensure that feature importances are correctly extracted and correspond to features
                if hasattr(model, 'feature_importances_'):
                    feature_importances = model.feature_importances_
                    assert len(feature_importances) == len(X.columns), "Mismatch in feature importance and feature columns length"
                else: 
                    feature_importances = None
                results[model_name] = {
                    'report': classification_report(y_test, y_pred),
                    'model': model,
                    'feature_importance': feature_importances
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