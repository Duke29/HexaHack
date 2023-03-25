from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():

    input_text = request.form['text']
    input_features = cv.transform([input_text])
    prediction = model.predict(input_features)[0]
    if prediction == "ham":
        return "The input message is not spam."
    elif prediction== "spam":
        return "The input message is spam."
    elif prediction=="spamservice":
        return "The input message is Service related spam"
    elif prediction=="spam18":
        return "The input message is adult content related spam"
if __name__ == '__main__':
    app.run(debug=True)