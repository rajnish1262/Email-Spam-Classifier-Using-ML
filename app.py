import streamlit as st
from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# Load the dataset
data = pd.read_csv('spam.csv')
data = data[['v1', 'v2']]  # Adjust column names based on your CSV
data.columns = ['label', 'message']

# Encode labels: ham = 0, spam = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Convert text data to numerical data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Make predictions
y_pred = classifier.predict(X_test_counts)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)


def transform_text(message):
    message_count = vectorizer.transform([message])
    prediction = classifier.predict(message_count)
    return 'Spam' if prediction[0] == 1 else 'Ham'


st.title("Email spam Detection with Machine Learning")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    st.header(transform_text(input_sms))