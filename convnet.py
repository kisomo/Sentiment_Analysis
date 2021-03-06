# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import string
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils as np_utils
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.layers import Embedding
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
import pickle
#import cPickle
#import dill

df = pd.read_csv('/home/terrence/CODING/Python/MODELS/Sentiment_Analysis/fake_or_real_news.csv').sample(1500)
print(df.shape)
print(df.head(3))
df1 = df

X = df['text']
df['label_num'] = df.label.map({'FAKE':0,'REAL':1})
y = df.label_num

MAX_NB_WORDS = 80000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 300

GLOVE_PATH = '/home/terrence/CODING/Python/MODELS/glove.6B.300d.txt'
encoding = "utf-8"

embeddings_index = {}
#f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
f = open(GLOVE_PATH, "rb")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#https://stackoverflow.com/questions/43076609/how-to-calculate-precision-and-recall-in-keras

def as_keras_metric(method):
    import functools
    from keras import backend as K
    #import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)

texts = df['text']
labels = df['label_num']

print('Found %s texts :' % len(texts))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print(type(sequences))
print(sequences[:5])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np_utils.to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

print(data[:3,:])
print(labels[:3])

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


#embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
#preds = Dense(len(labels_index), activation='softmax')(x)
#preds = Dense(len(labels), activation='softmax')(x)
#preds = Dense(2, activation='softmax')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc',precision,recall])

# happy learning!
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=1, batch_size=128)



'''
save_classifier = open("keras_fakenews_model.pickle", "wb")
pickle.dump(model, save_classifier)
#cPickle.dump(model, save_classifier)
##dill.dump(model, save_classifier)
save_classifier.close()
print("hoora!")


classifier_f = open("keras_fakenews_model.pickle","rb")
model = pickle.load(classifier_f)
classifier_f.close()

y_pred = model.predict(X_val)
print(y_pred)
#print((y_pred != y_val).sum())
#print(y_test)
#score = model.evaluate(x_val, y_val, verbose =0)
#print(score)

compl = np.array(['this is want I was telling him'])

def predict_complaint(complaint):
    #takes complaint
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(compl)
    comp2 = tokenizer.texts_to_sequences(compl)

    comp_index = tokenizer.word_index
    #dt = pad_sequences(comp2, maxlen=MAX_SEQUENCE_LENGTH)

    #comp_vector = np.zeros((len(comp_index) + 1, EMBEDDING_DIM))
    comp_vector = np.zeros((len(comp_index) + 1, MAX_SEQUENCE_LENGTH))
    for word, i in comp_index.items():
        embed_vector = embeddings_index.get(word)
        if embed_vector is not None:
            # words not found in embedding index will be all-zeros.
            #comp_vector[i] = embed_vector
            comp_vector[i,:EMBEDDING_DIM] = embed_vector
            ##comp_vector[i] = pad_sequences(comp_vector[i], maxlen=MAX_SEQUENCE_LENGTH)
            t = np.mean(comp_vector, axis=0) #comp_vector
            t1 = t.reshape(-1,1000)
    return t1 #np.mean(comp_vector, axis=0) #comp_vector

res = predict_complaint(compl)
print(res.shape)
#print(model.predict(res))

'''




