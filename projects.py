import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Embeddings
#https://embeddings.macheads101.com/


##++++++++++++++++++++++++++++++++++++++++fake news datacamp +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''

df = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv")
print(df.shape)
print(df.head(3))
##df.to_csv('fake_or_real_news.csv') 

#df = pd.read_csv("fake_or_real_news.csv")

#print(df.shape)
#print(df.head(3))

#print("\n\n")
#print(df.columns.values)

#df1 = df['Unnamed: 0.1', 'title', 'text', 'label']

#print(df1.shape)
#print(df1.head(3))

#print("\n\n")
#print(df1.columns.values)


df = df.set_index("Unnamed: 0")
print(df.head(3))

y = df.label
df.drop("label", axis =1)

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test, y_train, y_test = train_test_split(df['text'],y, test_size=0.33, random_state=53)


count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.fit_transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english',max_df=0.7)
#tfidf_vectorizer = TfidfTransformer(stop_words = 'english',max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.fit_transform(X_test)


print(tfidf_vectorizer.get_feature_names()[-10:])

print(count_vectorizer.get_feature_names()[:10])


# done with data preparation now create the confusion matrix function

def plot_confusion_matrix(cm, classes, normalizer=False, title = 'Confusion Matrix', cmap =plt.cm.Blues):
    
    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arrange(len(classes))
    plt.xticks(tick_marks, classes, rotation =45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    else:
        print('Confusion matrix, Without normalization')
    
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j], horizontalalignment='center',color='white' if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#Multinomial Naive Bayes with countvectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf = MultinomialNB()

clf.fit(tfidf_train,y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test,pred)
print("accuracy:  %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
plot_confusion_matrix(cm, classes = ['FAKE','REAL'])

'''






#++++++++++++++++++++++++++++++++++++ pre-trained glove or sefl train word2vec ++++++++++++++++++++++++++++++++++++++++++++++



#http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/


#wget http://nlp.stanford.edu/data/glove.6B.zip
#unzip glove.6B.zip

#and use load them up in python:

import numpy as np


#with open("glove.6B.50d.txt", "rb") as lines:
with open("/home/terrence/CODING/Python/MODELS/glove.6B.50d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}

#or we can train a Word2Vec model from scratch with gensim:

import gensim
## let X be a list of tokenized texts (i.e. list of lists of tokens)
#model = gensim.models.Word2Vec(X, size=100)
#w2v = dict(zip(model.wv.index2word, model.wv.syn0))


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])



#Let us throw in a version that uses tf-idf weighting scheme for good measure

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

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


from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])











#+++++++++++++++++++++++++++++++++++++++++++++++++organizing emails +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#http://www.andreykurenkov.com/writing/ai/organizing-my-emails-with-a-neural-net/



















#+++++++++++++++++++++++++++++++++++++++++++++++++++ Toxic comments +++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://www.kaggle.com/emannueloc/using-word-embeddings-with-gensim

import pandas as pd
df = pd.read_csv('train.csv')
corpus_text = '\n'.join(df[:5000]['comment_text'])
sentences = corpus_text.split('\n')
sentences = [line.lower().split(' ') for line in sentences]

def clean(s):
    return [w.strip(',."!?:;()\'') for w in s]
sentences = [clean(s) for s in sentences if len(s) > 0]

from gensim.models import Word2Vec

model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=4)

vectors = model.wv
del model


vectors['good']

print(vectors.similarity('you', 'your'))
print(vectors.similarity('you', 'internet'))


vectors.most_similar('i')




















#pre-trained glove
#https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python















#glove bots
#https://medium.com/rasa-blog/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d

















#pre-trained office automation
#https://www.youtube.com/watch?v=hEE8ZxcXxu4&feature=youtu.be







