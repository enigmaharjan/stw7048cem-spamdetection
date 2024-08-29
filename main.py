from flask import Flask, request, render_template
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Function to process input text
def process_input(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

# Load the saved vectorizer and model
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))

# Create the Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the email content from the form
        email_content = request.form['email_content']
        
        # Process and transform the input text
        processed_input = vectorizer.transform([email_content])

        # Predict using the loaded model
        prediction = model.predict(processed_input)

        # Determine the result
        if prediction[0] == 1:
            result = "The email is classified as spam."
        else:
            result = "The email is not spam."

        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
