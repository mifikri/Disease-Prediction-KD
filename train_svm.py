# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.svm import LinearSVC

df = pd.read_csv('./dataset/clean_data.csv')

X_train, X_test, y_train, y_test = train_test_split(df['Symptom'],df['Disease'], test_size=0, random_state = 0)
print type(X_train[0]), X_train[0]
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

def tokenizer(x):
    X_train_counts = count_vect.fit_transform(x)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf

def tokenizerTest(xTest):
    X_test_counts = count_vect.transform(xTest)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    return X_test_tfidf

#print len(count_vect.vocabulary_)
clf = LinearSVC().fit(tokenizer(X_train), y_train)
diseasePredict =  clf.predict(tokenizer(X_train))

filename = 'model_specialist.sav'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = clf.score(tokenizer(X_train), y_train)
print(result)

#example of input test 
sym = [['snuffle', 'throat sore','scleral icterus', 'nasal flaring', 'dysuria, lip smacking', 'headache', 'malaise', 'debilitation', 'symptom aggravating factors', 'chill', 'uncoordination', 'fever', 'pleuritic pain', 'sneeze', 'snore', 'green sputum', 'shortness of breath', 'distress respiratory', 'blackout', 'extreme exhaustion']]

#should merge each string element "[uncoordination, fever, pleuritic pain, snuffle, throat sore, malaise, debilitation, symptom aggravating factors, chill, scleral icterus, nasal flaring, dysuria, lip smacking, headache, sneeze, snore, green sputum, shortness of breath, distress respiratory, blackout, extreme exhaustion]"
symptomTest = []
for i in range (len(sym)):
    row_symptoms = " ".join(sym[i])
    symptomTest.append(row_symptoms)

print(clf.predict(count_vect.transform(symptomTest)))
print(loaded_model.predict(tokenizerTest(symptomTest)))

