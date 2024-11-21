import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk 
from nltk import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("info.json").read())

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

#print(documents)

words =[lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0] #because doc[0] = pattern and doc[1] = tag
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(output_empty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)

training = np.array(training)

train_X = training[: , :len(words)]
train_Y = training[: , len(words):]



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128 , input_shape = (len(train_X[0]),),activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64 , activation = 'relu'))
model.add(tf.keras.layers.Dense(len(train_Y[0]),activation ='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate = 0.1 , momentum = 0.9 , nesterov =True)

model.compile(loss ='categorical_crossentropy',optimizer = sgd , metrics = ['accuracy'])

hist = model.fit(np.array(train_X),np.array(train_Y),epochs=200 , batch_size = 5 , verbose = 1 )

model.save('chatbot.h5',hist)