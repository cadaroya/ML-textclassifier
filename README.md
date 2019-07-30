# ML-textclassifier

This project features RESTful webapp that uses Machine Learning (ML) to classify a block of text into categories (Tech, Entertainment, Politics, Sports, Business). 

## Overview
* Uses Flask framework and Gunicorn
* Uses TFIDFVectorizer to build feature vectors
* MultinomialNB for creating the predictive model
* Uses pickles to serialize/deserialize created models
* Basic HTML/CSS/JS for the front-end
* Deployed in heroku. Access it here: `https://text-classifier-app.herokuapp.com/`
* Dataset used is here: `https://docs.google.com/spreadsheets/d/1M89baUPgXGDvuyFdCwCit199t05g7jjncsxa02EattQ/edit?usp=sharing`


## Setup
1. `pip install -r requirements.txt`
2. `py app.py`
3. Open browser then access `localhost:5000`

## Model creation: senti.py
The program is pretty straightforward, following basic ML workflow (1-Preprocessing/Cleaning, 2-Feature Extraction, 3-Model Training, 4-Evaluation)

### 1. Preprocessing

* Checking data balance/imbalance
Firstly, the number of items per category was counted to check whether Dataset was balanced/imbalanced. Here are the values:

`'sport': 511, 'business': 510, 'politics': 417, 'tech': 401, 'entertainment': 386`.

Fortunately, items were fairly distributed per category so no modifications (oversampling/undersampling) were done.

We now proceed to cleaning to remove irrelevant data (those that won't help in building the model). 

* Removal of Special OS dependent characters

Characters like `\r`, `\n`, and other whitespaces were removed.

* Lowercasing

Dataset was already lowercased so no modifications were made.

* Removal of Punctuations

Punctuations didn't really help much in classifying categories, so they were removed. Samples of punctuations removed are as follows: `?`,`:`,`!`,`.`,`,`,`;`

* Stemming and Removal of Stopwords

Used NLTK to gain access to stemming and stopwords dictionaries. Using this, stopwords from the dataset were removed. Afterwards, every word was stemmed.

* Removal of non-alphabetic characters
Only words were considered to have effect on a text's category. As such, non-alphabetic characters were removed. 


### 2. Feature Building

Term Frequency-Inverse Document Frequence (TF-IDF) was used to numerically represent the 'text' items into features. The intuition in TF-IDF is that it increases the weight of a term if it is frequently found in a given document. 

### 3. Training

* Full Training Set and Split Training/Test

The model being used in the webapp uses the whole dataset as training set.
However, in order to properly evaluate the model, another model variable was created and used an 85-15 training-test ratio.

* Multinomial Naive Bayes

MultinomialNB (from scikit-learn) was used to create the classifier. It has a proven track-record for being used in problems that involves counting frequencies.
### 4. Evaluation

For metrics, `sklearn.metrics` was used. 

```
Accuracy on TRAINING set: 0.9725013220518244
Accuracy on TEST set: 0.9790419161676647
              precision    recall  f1-score   support

           0       0.98      0.96      0.97        56
           1       0.97      0.98      0.98        61
           2       0.99      0.99      0.99        77
           3       0.97      0.96      0.97        77
           4       0.98      1.00      0.99        63

   micro avg       0.98      0.98      0.98       334
   macro avg       0.98      0.98      0.98       334
weighted avg       0.98      0.98      0.98       334
```

As seen above, the score on the test set was `97.9041%` which was fairly accurate. However, it must be noted that the model only had a limited dictionary. Words that are unknown to the classifier will have 0 weight on the classification.

------

Afterwards, models from this run were pickled (serialized) so that it can be used in the RESTful API. 
