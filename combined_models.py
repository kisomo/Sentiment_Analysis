# -*- coding: utf-8 -*-

''' Text Classification using word2vec'''
#http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

#https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking_python3.ipynb

from tabulate import tabulate
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

# TRAIN_SET_PATH = "20ng-no-stop.txt"
# TRAIN_SET_PATH = "r52-all-terms.txt"
TRAIN_SET_PATH = "/home/terrence/CODING/Python/MODELS/r8-no-stop.txt"

GLOVE_6B_50D_PATH = "/home/terrence/CODING/Python/MODELS/glove.6B.50d.txt"
GLOVE_840B_300D_PATH = "/home/terrence/CODING/Python/MODELS/glove.6B.300d.txt"
encoding="utf-8"


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''

X, y = [], []
with open(TRAIN_SET_PATH, "r") as infile:
    for line in infile:
        label, text = line.split("\t")
        # texts are already tokenized, just split on space
        # in a real case we would use e.g. spaCy for tokenization
        # and maybe remove stopwords etc.
        X.append(text.split())
        y.append(label)
X, y = np.array(X), np.array(y)
print ("total examples %s" % len(y))
print(X.shape)
print(y.shape)
#print(X[0])
print(len(X[0]))
print(type(X))
print(y[0])
print(np.unique(y))


with open(GLOVE_6B_50D_PATH, "rb") as lines:
    wvec = {line.split()[0].decode(encoding): np.array(line.split()[1:],dtype=np.float32)
    #w2v = {line.split()[0].decode(encoding): np.array(line.split()[1:],dtype=np.float32)
               for line in lines}

print(type(wvec))
#print(wvec.keys())
print(len(wvec))


from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


n_items = take(2, wvec.keys())
#for index in wvec.keys():
for index in n_items:
    print(index, "==>", wvec[index])


# reading glove files, this may take a while
# we're reading line by line and only saving vectors
# that correspond to words from our training set
# if you wan't to play around with the vectors and have 
# enough RAM - remove the 'if' line and load everything

#import struct 
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
glove_small = {}
all_words = set(w for words in X for w in words)
print(len(all_words))
with open(GLOVE_6B_50D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode(encoding)
        if (word in all_words):
            nums=np.array(parts[1:], dtype=np.float32)
            glove_small[word] = nums

n_items = take(2, glove_small.keys())
print(len(glove_small))

for index in n_items:
    print(index, "==>", glove_small[index])


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")          
glove_big = {}
with open(GLOVE_840B_300D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode(encoding)
        if word in all_words:
            nums=np.array(parts[1:], dtype=np.float32)
            glove_big[word] = nums

n_items = take(2, glove_big.keys())
print(len(glove_big))

for index in n_items:
    print(index, "==>", glove_big[index])


print(">>>>>>>>>>>>>>>>>>train our own >>>>>>>>>>>>>>>>>>>>>>>")
## train word2vec on all the texts - both training and test set
## we're not using test labels, just texts so this is fine
#model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
#w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

model = Word2Vec(sentences=X, # tokenized senteces, list of list of strings
                 size=100,  # size of embedding vectors
                 window = 5, # I don't know what this means
                 workers=8, # how many threads?
                 min_count=5, # minimum frequency per token, filtering rare words
                 sample=0.05, # weight of downsampling common words
                 sg = 0, # should we use skip-gram? if 0, then cbow
                 iter=10,
                 hs = 0
        )


w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
print(type(w2v))
print(len(w2v))

maneno = model[model.wv.vocab]

print(maneno.shape)

print (model.most_similar('buy'))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print (model.most_similar('news'))
#print (model.most_similar('girl', number = 3))
#print(model.difference_in_hierarchy('mammal.n.01', 'dog.n.01'))
#print(model.difference_in_hierarchy('dog.n.01', 'mammal.n.01'))
#print(model.distance('mammal.n.01', 'carnivore.n.01'))
#print(model.distances('mammal.n.01', ['carnivore.n.01', 'dog.n.01']))
#print(model.distances('mammal.n.01'))
print(model.most_similar('work'))
#print(model.norm('work'))
print(model.similarity('work', 'news'))
#print(model.word_vec('office'))
#print(model.words_closer_than('work', 'news'))



n_items = take(2, w2v.keys())

for index in n_items:
    print(index, "==>", w2v[index])


# start with the classics - naive bayes of the multinomial and bernoulli varieties
# with either pure counts or tfidf features
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
# SVM - which is supposed to be more or less state of the art 
# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])

print(type(mult_nb))
#for index in mult_nb.keys():
#    print(index,mult_nb[index])

mult_nb.fit(X,y)
y1 = mult_nb.predict(X)
print(y1)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=400000 #len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

est1 = MeanEmbeddingVectorizer(glove_small)
print(type(est1))
print(est1.fit(X,y))
print(est1.transform(X))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
est2 = MeanEmbeddingVectorizer(glove_big)
print(type(est2))
print(est2.fit(X,y))
print(est2.transform(X))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
est3 = MeanEmbeddingVectorizer(w2v)
print(type(est3))
print(est3.fit(X,y))
print(est3.transform(X))


# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=400000 #len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


estim1 = TfidfEmbeddingVectorizer(glove_small)
print(type(estim1))
print(estim1.fit(X,y))
print(estim1.transform(X))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
estim2 = TfidfEmbeddingVectorizer(glove_big)
print(type(estim2))
print(estim2.fit(X,y))
print(estim2.transform(X))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
estim3 = TfidfEmbeddingVectorizer(w2v)
print(type(estim3))
print(estim3.fit(X,y))
print(estim3.transform(X))



# Extra Trees classifier is almost universally great, let's stack it with our embeddings
etree_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_big = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_big)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_big_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_big)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])


all_models = [
    ("mult_nb", mult_nb),
    ("mult_nb_tfidf", mult_nb_tfidf),
    ("bern_nb", bern_nb),
    ("bern_nb_tfidf", bern_nb_tfidf),
    ("svc", svc),
    ("svc_tfidf", svc_tfidf),
    ("w2v", etree_w2v),
    ("w2v_tfidf", etree_w2v_tfidf),
    ("glove_small", etree_glove_small),
    ("glove_small_tfidf", etree_glove_small_tfidf),
    ("glove_big", etree_glove_big),
    ("glove_big_tfidf", etree_glove_big_tfidf),

]
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(cross_val_score(mult_nb,X,y,cv=3)) #.mean()

unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])


print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))


plt.figure(figsize=(15, 6))
sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])
plt.show()



def benchmark(model, X, y, n):
    test_size = 1 - (n / float(len(y)))
    scores = []
    for train, test in StratifiedShuffleSplit(y, n_iter=5, test_size=test_size):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
    return np.mean(scores)


train_sizes = [10, 40, 160, 640, 3200, 6400]
table = []
for name, model in all_models:
    for n in train_sizes:
        table.append({'model': name, 
                      'accuracy': benchmark(model, X, y, n), 
                      'train_size': n})
df = pd.DataFrame(table)

plt.figure(figsize=(15, 6))
fig = sns.pointplot(x='train_size', y='accuracy', hue='model', 
                    data=df[df.model.map(lambda x: x in ["mult_nb", "svc_tfidf",# "w2v_tfidf", 
                                                         "glove_small_tfidf", "glove_big_tfidf", 
                                                        ])])
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="accuracy")
fig.set(xlabel="labeled training examples")
fig.set(title="R8 benchmark")
fig.set(ylabel="accuracy")

'''


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''
t_comments_train = pd.read_csv('/home/terrence/CODING/Python/MODELS/Sentiment_Analysis/train.csv').sample(2000)

print(t_comments_train.shape)
print(t_comments_train.columns)
print('------------------------------------')
X1 = t_comments_train.iloc[:,1]
y1 = t_comments_train.iloc[:,2]
print(X1.head(3))
print("------------------------------------")
print(y1.head(3))

import re
import string
#re_tok = re.compile(r'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
re_tok = re.compile(r'([{string.punctuation}“”¨«»®´·!\:/()<>=+#[]{}|º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r'\1', s).split()

X1 = X1.apply(lambda comment: tokenize(comment))
print(X1.head(3))

#X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=0)

from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))



#with open(GLOVE_6B_50D_PATH, "rb") as lines:
#    wvec = {line.split()[0].decode(encoding): np.array(line.split()[1:],dtype=np.float32)
#    #w2v = {line.split()[0].decode(encoding): np.array(line.split()[1:],dtype=np.float32)
#               for line in lines}

#print(type(wvec))

#n_items = take(2, wvec.keys())
#for index in n_items:
#    print(index, "==>", wvec[index])


# reading glove files, this may take a while
# we're reading line by line and only saving vectors
# that correspond to words from our training set
# if you wan't to play around with the vectors and have 
# enough RAM - remove the 'if' line and load everything

import struct 

glove_small = {}
all_words = set(w for words in X1 for w in words)
with open(GLOVE_6B_50D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode(encoding)
        if (word in all_words):
            nums=np.array(parts[1:], dtype=np.float32)
            glove_small[word] = nums


n_items = take(2, glove_small.keys())
for index in n_items:
    print(index, "==>", glove_small[index])

            
glove_big = {}
with open(GLOVE_840B_300D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode(encoding)
        if word in all_words:
            nums=np.array(parts[1:], dtype=np.float32)
            glove_big[word] = nums

n_items = take(2, glove_big.keys())
for index in n_items:
    print(index, "==>", glove_big[index])

print(len(all_words))


## train word2vec on all the texts - both training and test set
## we're not using test labels, just texts so this is fine
#model = Word2Vec(X1, size=100, window=5, min_count=5, workers=2)
#w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

model = Word2Vec(sentences=X1, # tokenized senteces, list of list of strings
                 size=100,  # size of embedding vectors
                 window = 5, # I don't know what this means
                 workers=8, # how many threads?
                 min_count=5, # minimum frequency per token, filtering rare words
                 sample=0.05, # weight of downsampling common words
                 sg = 0, # should we use skip-gram? if 0, then cbow
                 iter=10,
                 hs = 0
        )


w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
print(type(w2v))
print(len(w2v))

maneno = model[model.wv.vocab]

print(maneno.shape)

print (model.most_similar('fuck'))



# start with the classics - naive bayes of the multinomial and bernoulli varieties
# with either pure counts or tfidf features
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
# SVM - which is supposed to be more or less state of the art 
# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=400000 #len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
            
    def fit(self, X1, y1):
        return self 

    def transform(self, X1):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X1
        ])

est1 = MeanEmbeddingVectorizer(glove_small)
print(type(est1))
print(est1.fit(X1,y1))
print(est1.transform(X1))


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
est2 = MeanEmbeddingVectorizer(glove_big)
print(type(est2))
print(est2.fit(X1,y1))
print(est2.transform(X1))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
est3 = MeanEmbeddingVectorizer(w2v)
print(type(est3))
print(est3.fit(X1,y1))
print(est3.transform(X1))

 
# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=400000 #len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
        
    def fit(self, X1, y1):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X1)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X1):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X1
            ])


estim1 = TfidfEmbeddingVectorizer(glove_small)
print(type(estim1))
print(estim1.fit(X1,y1))
print(estim1.transform(X1))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
estim2 = TfidfEmbeddingVectorizer(glove_big)
print(type(estim2))
print(estim2.fit(X1,y1))
print(estim2.transform(X1))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
estim3 = TfidfEmbeddingVectorizer(w2v)
print(type(estim3))
print(estim3.fit(X1,y1))
print(estim3.transform(X1))


# Extra Trees classifier is almost universally great, let's stack it with our embeddings
etree_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_big = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_big)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_big_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_big)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])



all_models = [
    ("mult_nb", mult_nb),
    ("mult_nb_tfidf", mult_nb_tfidf),
    ("bern_nb", bern_nb),
    ("bern_nb_tfidf", bern_nb_tfidf),
    ("svc", svc),
    ("svc_tfidf", svc_tfidf),
    ("w2v", etree_w2v),
    ("w2v_tfidf", etree_w2v_tfidf),
    ("glove_small", etree_glove_small),
    ("glove_small_tfidf", etree_glove_small_tfidf),
    ("glove_big", etree_glove_big),
    ("glove_big_tfidf", etree_glove_big_tfidf),

]


unsorted_scores = [(name, cross_val_score(model, X1, y1, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])


print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))


plt.figure(figsize=(15, 6))
sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])
plt.show()

'''


'''
def benchmark(model, X1, y1, n):
    test_size = 1 - (n / float(len(y1)))
    scores = []
    for train, test in StratifiedShuffleSplit(y1, n_iter=5, test_size=test_size):
        X_train, X_test = X1[train], X1[test]
        y_train, y_test = y1[train], y1[test]
        scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
    return np.mean(scores)


train_sizes = [10, 40, 160, 640, 3200, 6400]
table = []
for name, model in all_models:
    for n in train_sizes:
        table.append({'model': name, 
                      'accuracy': benchmark(model, X1, y1, n), 
                      'train_size': n})
df = pd.DataFrame(table)

plt.figure(figsize=(15, 6))
fig = sns.pointplot(x='train_size', y='accuracy', hue='model', 
                    data=df[df.model.map(lambda x: x in ["mult_nb", "svc_tfidf", "w2v_tfidf", 
                                                         "glove_small_tfidf", "glove_big_tfidf", 
                                                        ])])
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="accuracy")
fig.set(xlabel="labeled training examples")
fig.set(title="R8 benchmark")
fig.set(ylabel="accuracy")
'''


#https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
'''
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

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

'''


#https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# load the dataset
data = open('/home/terrence/CODING/Python/MODELS/Amazon_Reviews').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(content[1])

# create a dataframe using texts and lables
trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

print(trainDF.shape)
print(trainDF.head(3))
print(np.unique(labels))

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

print(train_y.shape)
print(np.unique(train_y))

#------------------ countvectorizer ----------------------------------
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)



#------------------ tfidf word-level ----------------------------------
#TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)


#------------------ tfidf n-gram level ----------------------------------
# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


#------------------ tfidf character level ----------------------------------

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 



#------------------ Embeddings as features ---------------------------------

# load the pre-trained word-embedding vectors 
embeddings_index = {}
#https://fasttext.cc/docs/en/english-vectors.html
for i, line in enumerate(open('/home/terrence/CODING/Python/MODELS/wiki-news-300d-1M.vec')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

print(type(embeddings_index))
print(len(embeddings_index))


# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

print(type(word_index))

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)


# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



#-------------------- Text / NLP based features -----------------------------

trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}


# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))






#---------------------- Topic Models as features -------------------------------

# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

#models --------

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


#----------------- Model - Naive Bayes -------------------------

# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print "NB, Count Vectors: ", accuracy

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print "NB, WordLevel TF-IDF: ", accuracy

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print "NB, N-Gram Vectors: ", accuracy

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print "NB, CharLevel Vectors: ", accuracy
#NB, Count Vectors:  0.7004
#NB, WordLevel TF-IDF:  0.7024
#NB, N-Gram Vectors:  0.5344
#NB, CharLevel Vectors:  0.6872



#----------------- Model - Linear Classifier -------------------------

#----------------- Model - SVM -------------------------

#----------------- Model - Baggin -------------------------

#----------------- Model - Boosting -------------------------

#----------------- Model - Shallow Neural Nets -------------------------

#----------------- Model - CNN -------------------------

#----------------- Model - LSTM -------------------------

#----------------- Model - GRU -------------------------

#----------------- Model - Bidirectional RNN -------------------------

#----------------- Model - Recurrent CNN (RCNN) -------------------------

#----------------- Model - Other Deep Learning -------------------------




#https://github.com/cahya-wirawan/ML-Collection/blob/master/TextClassification.py




#https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa



#https://www.analyticsvidhya.com/blog/2017/07/word-representations-text-classification-using-fasttext-nlp-facebook/






#http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/




#http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/





#https://www.kaggle.com/marijakekic/cnn-in-keras-with-pretrained-word2vec-weights








