
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

print("#=======================================================================================================================================================================")


import gensim
from sklearn.feature_extraction.text import CountVectorizer
 
documents = sms #pd.read_csv('news-data.csv', error_bad_lines=False);
 
documents.head()






