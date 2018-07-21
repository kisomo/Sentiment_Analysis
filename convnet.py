# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import string

df = pd.read_csv('/home/terrence/CODING/Python/MODELS/Sentiment_Analysis/fake_or_real_news.csv')
print(df.shape)
print(df.head(3))
df1 = df

X = df['text']
df['label_num'] = df.label.map({'FAKE':0,'REAL':1})
y = df.label_num

'''
re_tok = re.compile(r'([{string.punctuation}“”¨«»®´·!\:/?()<>=+#[]{}|º½¾¿¡§£₤‘’])')

def tokenize(s):
     return re_tok.sub(r'\1', s).split()

X = X.apply(lambda comment: tokenize(comment))
print(X.head(3))

X,y = np.array(X), np.array(y)
'''

texts = df['text']
labels = df['label_num']

print('Found %s texts :' % len(texts))

MAX_NB_WORDS = 80000
MAX_SEQUENCE_LENGTH = 1000
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300

GLOVE_PATH = '/home/terrence/CODING/Python/MODELS/glove.6B.300d.txt'
encoding = "utf-8"

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils as np_utils
import os

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
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
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
#f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
f = open(GLOVE_PATH, "rb")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)



from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model

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
preds = Dense(len(labels), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=2, batch_size=128)















#X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=0)

