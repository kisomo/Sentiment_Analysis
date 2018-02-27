import numpy as np
# imports
#%matplotlib inline

#https://www.datascience.com/resources/notebooks/word-embeddings-in-python

import os
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import string
import re
from gensim import corpora
from gensim.models import Phrases
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#from ds_voc.text_processing import TextProcessing

raw_df = pd.read_csv("/home/terrence/CODING/Python/MODELS/Reviews.csv")
print(raw_df.shape)
print(raw_df.head(3))
# sample for speed
raw_df2 = raw_df.sample(frac=0.1,  replace=False)
print(raw_df2.shape)


# grab review text
#raw = list(raw_df2['Text'])
raw = raw_df2['Text'] #TERRENCE
raw = np.array(raw)
#print(raw)
print(len(raw))





#+++++++++++++++++++++++++++++++++++++++++ CountVectorizer ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#http://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/

from sklearn.feature_extraction.text import CountVectorizer

# 2. instantiate CountVectorizer (vectorizer)
vect = CountVectorizer()

vect.fit(raw)


sentences = vect.get_feature_names()
#print(sentences)

from gensim.models import Word2Vec

model = Word2Vec(sentences=sentences, # tokenized senteces, list of list of strings
                 size=300,  # size of embedding vectors
                 workers=8, # how many threads?
                 min_count=20, # minimum frequency per token, filtering rare words
                 sample=0.05, # weight of downsampling common words
                 sg = 0, # should we use skip-gram? if 0, then cbow
                 iter=5,
                 hs = 0
        )

X = model[model.wv.vocab]

print(X.shape)
#print (model.most_similar('peanut'))
#print (model.most_similar('coffee'))
#print (model.most_similar('spice'))
#print (model.most_similar(['snack', 'protein'], negative=['supplement']))


# visualize food data
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.rcParams['figure.figsize'] = [10, 10]
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()


from bokeh.plotting import figure, show
from bokeh.io import push_notebook, output_notebook
from bokeh.models import ColumnDataSource, LabelSet
import IPython #TERRENCE

def interactive_tsne(text_labels, tsne_array):
    #makes an interactive scatter plot with text labels for each point

    # define a dataframe to be used by bokeh context
    bokeh_df = pd.DataFrame(tsne_array, text_labels, columns=['x','y'])
    bokeh_df['text_labels'] = bokeh_df.index

    # interactive controls to include to the plot
    TOOLS="hover, zoom_in, zoom_out, box_zoom, undo, redo, reset, box_select"

    p = figure(tools=TOOLS, plot_width=700, plot_height=700)

    # define data source for the plot
    source = ColumnDataSource(bokeh_df)

    # scatter plot
    p.scatter('x', 'y', source=source, fill_alpha=0.6,
              fill_color="#8724B5",
              line_color=None)

    # text labels
    labels = LabelSet(x='x', y='y', text='text_labels', y_offset=8,
                      text_font_size="8pt", text_color="#555555",
                      source=source, text_align='center')

    p.add_layout(labels)

    # show plot inline
    output_notebook()
    show(p)
    

interactive_tsne(model.wv.vocab.keys(), X_tsne)

#+++++++++++++++++++++++++++++++++++++++++++++++ PRE-TRAINED word2vec ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#http://textminingonline.com/getting-started-with-word2vec-and-glove-in-python
##By gensim word2vec module, you can also load the trained model by original c/c++ word2vec pakcage:
print("++++++++++++++++++++++++++++ pre-trained word2vec +++++++++++++++++++++++++++++++++++++++++++\n\n\n")
corpus = [
          'Text of the first document.',
          'Text of the second document made longer.',
          'Number three.',
          'This is number four.',
]
## we need to pass splitted sentences to the model
tokenized_sentences = [sentence.split() for sentence in corpus]
from gensim.models import KeyedVectors #.load_word2vec_format

##model_org = word2vec.Word2Vec.load_word2vec_format('vectors.bin', binary=True)

##model_org = KeyedVectors.load_word2vec_format('vectors.bin', binary=True)
#model_org = KeyedVectors.load_word2vec_format(tokenized_sentences, binary=True)
#print(model_org.most_similar('text'))


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ glove +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#http://textminingonline.com/getting-started-with-word2vec-and-glove-in-python
'''
import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove
sentences = list(itertools.islice(Text8Corpus('text8'),None))
corpus = Corpus()
corpus.fit(sentences, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
#Performing 30 training epochs with 4 threads

glove.add_dictionary(corpus.dictionary)
 
glove.most_similar('man')

glove.most_similar('man', number=10)

 
glove.most_similar('frog', number=10)

 
glove.most_similar('girl', number=10)

glove.most_similar('car', number=10)

 
glove.most_similar('queen', number=10)
'''


#+++++++++++++++++++++++++++++++++++++++ toy glove ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://github.com/maciejkula/glove-python

from glove import Glove
from glove import Corpus

raw = list(np.array(raw_df2['Text']))

corpus_model = Corpus()
#corpus_model.fit(get_data(args.create), window=10)
corpus_model.fit(raw, window=10)
corpus_model.save('corpus.model')
        
print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=int(args.train), no_threads=args.parallelism, verbose=True)
glove.add_dictionary(corpus_model.dictionary)

glove.save('glove.model')

print(glove.most_similar('book', number=10))


