import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.metrics import accuracy_score

# Set page title and credits
st.title('Sentiment Analysis of Movie Reviews using Machine Learning techniques')
st.subheader('Supervised by: DR, Ben Ahmed Mohamed')
st.markdown('Prepared by:')
st.markdown('- AOUFI AYMAN')
st.markdown('- BELLAHSINI Anass')
st.markdown('- BERROUHOU Aissa')
st.markdown('- GHAFIR ADIL')

# Load the data
data = pd.read_csv('Valid.csv')

# Specify the correct column names for the features (X) and target variable (y)
feature_column = 'text'
target_column = 'label'

# Check if the target column exists in the DataFrame
if target_column not in data.columns:
    raise KeyError(f"'{target_column}' not found in the DataFrame.")

# Separate the features (X) and target variable (y)
X = data[feature_column]
y = data[target_column]

# Create an imputer and fit it on the target variable
imputer = SimpleImputer(strategy='mean')
y = imputer.fit_transform(y.values.reshape(-1, 1))

# Create a TF-IDF vectorizer and transform the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Create and fit the logistic regression classifier
logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)

# Save the trained models using pickle
with open('models.pkl', 'wb') as file:
    pickle.dump((logistic_regression, vectorizer), file)

# Streamlit app code
st.header('Logistic regression')

# Input text
input_text = st.text_area('Enter the text to classify:', '')

# Classification and display results
if st.button('Classify'):
    # Load the saved models
    with open('models.pkl', 'rb') as file:
        logistic_regression, vectorizer = pickle.load(file)

    # Perform classification on the input text
    if input_text:
        transformed_text = vectorizer.transform([input_text])
        prediction = logistic_regression.predict(transformed_text)
        if prediction[0] == 1:
            st.write('Sentiment: Positive')
        else:
            st.write('Sentiment: Negative')

        # Load the reference dataset with true labels
        reference_data = pd.read_csv('Valid.csv')
        reference_X = reference_data[feature_column]
        reference_y = reference_data[target_column]

        # Transform reference dataset using the same vectorizer
        transformed_reference = vectorizer.transform(reference_X)

        # Predict sentiment on the reference dataset
        reference_prediction = logistic_regression.predict(transformed_reference)

        # Calculate quality score (accuracy) using true labels and predicted labels
        quality_score = accuracy_score(reference_y, reference_prediction)
        st.write('Quality Score:', quality_score)
