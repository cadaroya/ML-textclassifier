import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# Remove stopwords, stemmed
# Separate labels and training data
# 

def buildModels_training(df, X_train, y_train):
	# Vectorize training set
	max_features = 1000

	tfidf = TfidfVectorizer(encoding='utf-8',
	                        stop_words=None,
	                        lowercase=False,
	                        max_features=max_features,)

	# Fitting of training set (VECTORIZATION)
	features_train = tfidf.fit_transform(X_train).toarray()

	# Fitting of training set (MODEL)
	mnb = MultinomialNB()
	mnb.fit(features_train, y_train.astype('int'))


	# Pickling
	with open('TFIDF.pkl', 'wb') as handle:
		pickle.dump(tfidf, handle, pickle.HIGHEST_PROTOCOL)

	with open('MultinomialNB.pkl', 'wb') as handle:
		pickle.dump(mnb, handle, pickle.HIGHEST_PROTOCOL)

	return tfidf, mnb

	

def clean(df, labels):
	# Special Characters removal
	df['text'] = df['text'].str.replace("\r", " ")
	df['text'] = df['text'].str.replace("\n", " ")
	df['text'] = df['text'].str.replace("    ", " ")
	df['text'] = df['text'].str.replace('"', '')

	# Already lowercased
	# Punctuations
	punctuation_signs = list("?:!.,;")
	for punct_sign in punctuation_signs:
		df['text'] = df['text'].str.replace(punct_sign, '')

	# Stemmer and stopwords
	stopWords = set(stopwords.words('english'))
	stemmer = PorterStemmer()

	# Tokenize texts
	for i, val in enumerate(df['text']):
		cleaned = nltk.wordpunct_tokenize(val)
		cleaned = [word.lower() for word in cleaned if word.isalpha() and (word) not in stopWords]
		cleaned = [stemmer.stem(word) for word in cleaned]

		df['text'][i] = " ".join(cleaned)
		

	# Replace category with labels
	for i,c in enumerate(df['category']):
		df['category'][i] = labels[c]

	return df

# Read dataset.csv into panda probably
df = pd.read_csv('dataset.csv')

# Check imbalance/balance of dataset
counts = Counter(df['category'])
print(counts)

# Labels (For replacing category with numerical value)
labels = {"tech": 0, "politics": 1, "sport": 2, "business":3, "entertainment":4}
print(labels)

# Cleaning/Preprocessing of data
df = clean(df,labels)

# Split Data Set
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.15, random_state=8)

#print("________")
#print(X_test)
#print(y_test)
#print("________")

# Build vectors and models then PICKLE
tfidf, mnb = buildModels_training(df, X_train, y_train)

# Initial Predicting using test data
feature_test = tfidf.transform(X_test).toarray()
feature_train = tfidf.transform(X_train).toarray()
labels_train = y_train.astype('int')
labels_test = y_test.astype('int')
y_pred = mnb.predict(feature_test)
y_pred_train = mnb.predict(feature_train)

# Accuracies
print("Accuracy on TRAINING set: " + str(accuracy_score(labels_train, y_pred_train)))
print("Accuracy on TEST set: " +str(accuracy_score(labels_test, y_pred)))
# Classification
print(classification_report(labels_test, y_pred))


# In case whole dataset is for training data
features = tfidf.fit_transform(df['text']).toarray()
labels = df['category'].astype('int')
fullMNB = MultinomialNB()
fullMNB.fit(features, labels)

with open('fullMNB.pkl', 'wb') as handle:
	pickle.dump(fullMNB, handle, pickle.HIGHEST_PROTOCOL)

with open('fullTFIDF.pkl', 'wb') as handle:
    pickle.dump(tfidf, handle, pickle.HIGHEST_PROTOCOL)