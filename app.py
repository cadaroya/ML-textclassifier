import json
import logging
import pickle
import os
from flask import Flask,jsonify,request,render_template
from flask_cors import CORS
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)
CORS(app)
@app.route("/classify/<text>",methods=['GET'])
def classify(text):
	print("ENTERED")
	with open('pickles/fullMNB.pkl', 'rb') as handle:
		mnb = pickle.load(handle)

	with open('pickles/fullTFIDF.pkl', 'rb') as handle:
		tfidf = pickle.load(handle)

	feature = tfidf.transform([text.lower()]).toarray()
	labels = {0:"TECHNOLOGY", 1:"POLITICS", 2:"SPORTS", 3:"BUSINESS", 4:"ENTERTAINMENT"}

	
	val = mnb.predict(feature)
	
	response = {
		'category': labels[int(val[0])],
		'text': text,
	}

	return jsonify(response)

@app.route("/",methods=['GET'])
def default():
	return render_template('index.html')

if __name__ == "__main__":
	app.run(debug=True) 