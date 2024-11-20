import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk 
from nltk import WordNetLemmatizer

lemmatize = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignoreLetters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)