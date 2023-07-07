import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import pandas as pd

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
