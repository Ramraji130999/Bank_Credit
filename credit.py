import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv('application_record.csv')

# Separate features and target
X = data.drop(columns=['TARGET'])
y = data['TARGET']

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Optionally save preprocessor for future use
# joblib.dump(preprocessor, 'preprocessor.pkl')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Train a RandomForestClassifier (you can experiment with other models)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC Score: {roc_auc}')

# Optionally save the model for deployment
# joblib.dump(model, 'best_model.pkl')
import streamlit as st
import pandas as pd
import joblib

# Load preprocessor and model
# preprocessor = joblib.load('preprocessor.pkl')
# model = joblib.load('best_model.pkl')

# Function to preprocess user input
def preprocess_input(user_input):
    # Example function - modify based on actual preprocessing steps
    processed_input = user_input  # Replace with actual preprocessing steps
    return processed_input

# Streamlit app
st.title('Bank Risk Controller')

# Sidebar for user input
st.sidebar.header('User Input')

# Example input fields
user_input = {}
for column in X.columns:
    user_input[column] = st.sidebar.text_input(column, '')

# Preprocess user input
processed_input = preprocess_input(user_input)

# Example prediction
if st.sidebar.button('Predict'):
    # Example prediction using the model
    prediction = model.predict(processed_input)
    if prediction[0] == 1:
        st.write('Prediction: Default')
    else:
        st.write('Prediction: Not Default')

# Optionally add more sections for EDA, visualizations, etc.

