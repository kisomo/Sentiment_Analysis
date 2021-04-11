# -*- coding: utf-8 -*-

#sudo docker build . -t sentiment:1.0.0
#sudo docker images
#sudo docker run -d -p 5000:5000 sentiment:1.0.0
#sudo docker run --rm -it sentiment:1.0.0


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









