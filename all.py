import numpy as np

#https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795

#https://appliedmachinelearning.wordpress.com/2017/02/12/sentiment-analysis-using-tf-idf-weighting-pythonscikit-learn/

#++++++++++++++++++++++++++++++++++ CountVectorizing +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from sklearn.feature_extraction.text import CountVectorizer
# create CountVectorizer object
vectorizer = CountVectorizer()
corpus = [
          'Text of first document TEXT text.',
          'Text of the second document document document made longer.',
          'Number three three three three.',
          'This is number number number four.',
]
# learn the vocabulary and store CountVectorizer sparse matrix in X
#vectorizer.fit(corpus)
#tokenized_sentences = [sentence.split() for sentence in corpus]
#print(tokenized_sentences)

X = vectorizer.fit_transform(corpus)

print(X)
print("\n\n")
print(X.toarray())
print("\n\n")

# columns of X correspond to the result of this method
T = vectorizer.get_feature_names() == (
    ['document', 'first', 'four', 'is', 'longer',
     'made', 'number', 'of', 'second', 'text',
     'the', 'this', 'three'])

print(T)
print(vectorizer.get_feature_names())

# retrieving the matrix in the numpy form
X.toarray()
print(X.toarray())


print("+++++++++++++++++++++++++++++++++++++++++ TF-IDF +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

from sklearn.feature_extraction.text import TfidfTransformer
# create tf-idf object
transformer = TfidfTransformer(norm = 'l2', smooth_idf=True, sublinear_tf= False, use_idf =True )
# X can be obtained as X.toarray() from the previous snippet

# learn the vocabulary and store tf-idf sparse matrix in tfidf
tfidf = transformer.fit_transform(X)
print(tfidf)
# retrieving matrix in numpy form as we did it before
tfidf.toarray()    
print(tfidf.toarray() )     


##vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf =True, stop_words = 'english')

'''
print("+++++++++++++++++++++++++++++++++++++ word2vec - one word context ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

from gensim.models import word2vec
corpus = [
          'Text of the first document.',
          'Text of the second document made longer.',
          'Number three.',
          'This is number four.',
]

print(corpus)
# we need to pass splitted sentences to the model
tokenized_sentences = [sentence.split() for sentence in corpus]
print("\n\n")
print(tokenized_sentences)

#model = word2vec.Word2Vec(tokenized_sentences, min_count=1)

model = word2vec.Word2Vec(sentences=tokenized_sentences, # tokenized senteces, list of list of strings
                 size=3,  # size of embedding vectors
                 workers=8, # how many threads?
                 min_count=1, # minimum frequency per token, filtering rare words
                 sample=0.05, # weight of downsampling common words
                 sg = 1, # should we use skip-gram? if 0, then cbow
                 iter=5,
                 hs = 0
        )

X = model[model.wv.vocab]

print(X.shape)
print (model.most_similar('first'))
#print (model.most_similar('document'))
#print (model.most_similar('This'))
#print (model.most_similar(['snack', 'protein'], negative=['supplement']))

'''

'''
#+++++++++++++++++++++++++++++++++++++ word2vec - multi-word context +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from gensim.models import word2vec
corpus = [
          'Text of the first document.',
          'Text of the second document made longer.',
          'Number three.',
          'This is number four.',
]
# we need to pass splitted sentences to the model
tokenized_sentences = [sentence.split() for sentence in corpus]
model = word2vec.Word2Vec(tokenized_sentences, min_count=1)




#+++++++++++++++++++++++++++++++++++++ word2vec - skip-gram model +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from gensim.models import word2vec
corpus = [
          'Text of the first document.',
          'Text of the second document made longer.',
          'Number three.',
          'This is number four.',
]
# we need to pass splitted sentences to the model
tokenized_sentences = [sentence.split() for sentence in corpus]
model = word2vec.Word2Vec(tokenized_sentences, min_count=1)

'''

'''
print("+++++++++++++++++++++++++++++++++++++++ glove ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove

# sentences and corpus from standard library
#sentences = list(itertools.islice(Text8Corpus('text8'),None))
tokenized_sentences = [sentence.split() for sentence in corpus]
corpus = Corpus()

# fitting the corpus with sentences and creating Glove object
corpus.fit(tokenized_sentences, window=10)

glove = Glove(no_components=100, learning_rate=0.05)
# fitting to the corpus and adding standard dictionary to the object
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)




#++++++++++++++++++++++++++++++++++++++++++++++++++ FastText ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++







#++++++++++++++++++++++++++++++++++++++++++ Poincare metric +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''



