#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 21:29:10 2017

@author: alin
"""
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from BeautifulSoup import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

positive_reviews = BeautifulSoup(open('positive.review.txt').read())
positive_reviews = positive_reviews.findAll('review_text')
negative_reviews = BeautifulSoup(open('negative.review.txt').read())
negative_reviews = negative_reviews.findAll('review_text')


lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
stopwords.add('this')

t = positive_reviews[0].text

def my_tokenizer(s):                    
    tokens = word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2] 
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

N = len(positive_reviews) + len(negative_reviews)
data = np.zeros((N, len(word_index_map) + 1))

def token_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)
    for token in tokens:
        x[word_index_map[token]] = 1
    x = x/sum(x)
    x[-1] = label
    return x

i = 0
for tokens in positive_tokenized:
    xy = token_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1
#    
for tokens in negative_tokenized:
    xy = token_to_vector(tokens, -1)
    data[i,:] = xy
    i += 1
    
X = data[:,:-1]
y = data[:, -1]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 10, stratify = y)

lg = LogisticRegression()
lg.fit(X_train, y_train)
print lg.score(X_test, y_test)

rf = RandomForestClassifier(n_estimators = 500)
rf.fit(X_train, y_train)
print rf.score(X_test, y_test)
