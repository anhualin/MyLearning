#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 20:52:26 2017

@author: alin
"""
import nltk
nltk.download()
from nltk.examples.pt import *

nltk.corpus.mac_morpho.words()
nltk.corpus.mac_morpho.sents() 
nltk.corpus.mac_morpho.tagged_words()

from nltk.corpus import floresta
floresta.words()
floresta.tagged_words()
stok = nltk.data.load('tokenizers/punkt/portuguese.pickle')
stemmer = nltk.stem.RSLPStemmer()
stemmer.stem('copiar')
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords[:10]


from __future__ import division  # Python 2 users only
import nltk, re, pprint
from nltk import word_tokenize

f = open("/home/alin/MyLearning/EasyNLP/sample1.txt", "r")
raw = f.read()
tokens = word_tokenize(raw.decode('utf-8'))
text = nltk.Text(tokens)
