# train.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from text_processing import process

# Load your dataset
dataset = pd.read_csv('dataset/emails.csv')

# Create the vectorizer with the imported process function
vectorizer = CountVectorizer(analyzer=process)

# Fit the vectorizer to the dataset
message = vectorizer.fit_transform(dataset['text'])

# Create and train your model
model = MultinomialNB()
model.fit(message, dataset['spam'])

# Save the vectorizer
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

# Save the model
pickle.dump(model, open("models/model.pkl", "wb"))
