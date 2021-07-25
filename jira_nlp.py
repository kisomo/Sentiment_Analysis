
import sys
print(sys.path)


import numpy as np
import pandas as pd

#http://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/

#https://appliedmachinelearning.wordpress.com/2017/02/12/sentiment-analysis-using-tf-idf-weighting-pythonscikit-learn/

# alternative: read file into pandas from a URL
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_table(url, header=None, names=['label', 'message'])

# examine the shape
print(sms.shape)
#print(sms.head(3)) #TM

# examine the first 10 rows
print(sms.head(10))

# examine the class distribution
print(sms.label.value_counts())

# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})

# check that the conversion worked
print(sms.head())

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)

###############################################################################################################################################################################################
###############################################################################################################################################################################################
###############################################################################################################################################################################################

#https://predictivehacks.com/lda-topic-modelling-with-gensim/



print("#=======================================================================================================================================================================")

import gensim
from sklearn.feature_extraction.text import CountVectorizer
 
documents = sms #pd.read_csv('news-data.csv', error_bad_lines=False);
 
print(documents.head())


# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english',  token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(documents.message)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
 
# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())
 

# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`
 
ldamodel = gensim.models.LdaMulticore(corpus=corpus, id2word=id_map, passes=2, random_state=5, num_topics=10, workers=2)

#For each topic, we will explore the words occuring in that topic and its relative weight

for idx, topic in ldamodel.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")

#Let’s say that we want get the probability of a document to belong to each topic. Let’s take an arbitrary document from our data:

my_document = documents.message[17]
print(my_document)


def topic_distribution(string_input):
    string_input = [string_input]
    # Fit and transform
    X = vect.transform(string_input)
 
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
 
    output = list(ldamodel[corpus])[0]
 
    return output
  
 
print(topic_distribution(my_document))


#As we can see, this document is more likely to belong to topic 8 with a 51% probability. It makes sense because this document
# # is related to “war” since it contains the word “troops” and topic 8 is about war. Let’s recall topic 8:


#Let’s say that we want to assign the most likely topic to each document which is essentially the argmax of the distribution above.

def topic_prediction(my_document):
    string_input = [my_document]
    X = vect.transform(string_input)
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    output = list(ldamodel[corpus])[0]
    topics = sorted(output,key=lambda x:x[1],reverse=True)
    return topics[0][0]
 
print(topic_prediction(my_document))

print("----------------")

#print(topic_prediction(documents.message))

#That was an example of Topic Modelling with LDA. We could have used a TF-IDF instead of Bags of Words. Also, we could have applied 
# lemmatization and/or stemming. There are many different approaches. Our goal was to provide a walk-through example and feel free to try different approaches.




########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################

#https://www.kaggle.com/thebrownviking20/topic-modelling-with-spacy-and-scikit-learn


# Usual imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import concurrent.futures
import time
#import pyLDAvis.sklearn
#from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline
import os
#print(os.listdir("../input"))

# Plotly based imports for visualization
#from plotly import tools
#import plotly.plotly as py
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)
#import plotly.graph_objs as go
#import plotly.figure_factory as ff

# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
#!python3 -m spacy download en_core_web_lg



# Loading data
wines = sms #pd.read_csv('../input/winemag-data_first150k.csv')
wines.head()

# Creating a spaCy object
nlp = spacy.load('en_core_web_lg')

doc = nlp(wines["message"][3])
spacy.displacy.render(doc, style='ent',jupyter=True)

punctuations = string.punctuation
stopwords = list(STOP_WORDS)


review = str(" ".join([i.lemma_ for i in doc]))
print(review)

doc = nlp(review)
spacy.displacy.render(doc, style='ent',jupyter=True)

# POS tagging
for i in nlp(review):
    print(i,"=>",i.pos_)


# Parser for reviews
parser = English()
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


tqdm.pandas()
wines["processed_message"] = wines["message"] #wines["message"].progress_apply(spacy_tokenizer)

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(wines["processed_message"])


NUM_TOPICS = 10


# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)


# Non-Negative Matrix Factorization Model
nmf = NMF(n_components=NUM_TOPICS)
data_nmf = nmf.fit_transform(data_vectorized) 


# Latent Semantic Indexing Model using Truncated SVD
lsi = TruncatedSVD(n_components=NUM_TOPICS)
data_lsi = lsi.fit_transform(data_vectorized)


# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 


# Keywords for topics clustered by Latent Dirichlet Allocation
print("LDA Model:")
selected_topics(lda, vectorizer)


# Keywords for topics clustered by Latent Semantic Indexing
print("NMF Model:")
selected_topics(nmf, vectorizer)


# Keywords for topics clustered by Non-Negative Matrix Factorization
print("LSI Model:")
selected_topics(lsi, vectorizer)

# Transforming an individual sentence
text = spacy_tokenizer("Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity.")
x = lda.transform(vectorizer.transform([text]))[0]
print(x)

#pyLDAvis.enable_notebook()
#dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
#dash

svd_2d = TruncatedSVD(n_components=2)
data_2d = svd_2d.fit_transform(data_vectorized)
'''
trace = go.Scattergl(
    x = data_2d[:,0],
    y = data_2d[:,1],
    mode = 'markers',
    marker = dict(
        color = '#FFBAD2',
        line = dict(width = 1)
    ),
    text = vectorizer.get_feature_names(),
    hovertext = vectorizer.get_feature_names(),
    hoverinfo = 'text' 
)
data = [trace]
iplot(data, filename='scatter-mode')


trace = go.Scattergl(
    x = data_2d[:,0],
    y = data_2d[:,1],
    mode = 'text',
    marker = dict(
        color = '#FFBAD2',
        line = dict(width = 1)
    ),
    text = vectorizer.get_feature_names()
)
data = [trace]
iplot(data, filename='text-scatter-mode')


def spacy_bigram_tokenizer(phrase):
    doc = parser(phrase) # create spacy object
    token_not_noun = []
    notnoun_noun_list = []
    noun = ""

    for item in doc:
        if item.pos_ != "NOUN": # separate nouns and not nouns
            token_not_noun.append(item.text)
        if item.pos_ == "NOUN":
            noun = item.text
        
        for notnoun in token_not_noun:
            notnoun_noun_list.append(notnoun + " " + noun)

    return " ".join([i for i in notnoun_noun_list])



bivectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, ngram_range=(1,2))
bigram_vectorized = bivectorizer.fit_transform(wines["processed_description"])


bi_lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_bi_lda = bi_lda.fit_transform(bigram_vectorized)


print("Bi-LDA Model:")
selected_topics(bi_lda, bivectorizer)


bi_dash = pyLDAvis.sklearn.prepare(bi_lda, bigram_vectorized, bivectorizer, mds='tsne')
bi_dash

'''




########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################

#https://www.kaggle.com/dskswu/topic-modeling-bert-lda

'''

!pip install spacy-langdetect
!pip install language-detector
!pip install symspellpy
!pip install sentence-transformers



'''









