import numpy as np
import pandas as pd

#http://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/

#https://appliedmachinelearning.wordpress.com/2017/02/12/sentiment-analysis-using-tf-idf-weighting-pythonscikit-learn/



# alternative: read file into pandas from a URL
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_table(url, header=None, names=['label', 'message'])

# examine the shape
print(sms.shape)

# examine the first 10 rows
print(sms.head())

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

#X is 1D currently because it will be passed to Vectorizer to become a 2D matrix
#You must always have a 1D object so CountVectorizer can turn into a 2D object for the model to be built on

# split X and y into training and testing sets
# by default, it splits 75% training and 25% test
# random_state=1 for reproducibility
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#+++++++++++++++++++++++++++++++++ CountVectorizer +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1. import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer

# 2. instantiate CountVectorizer (vectorizer)
vect = CountVectorizer()

# learn training data vocabulary, then use it to create a document-term matrix

# 3. fit
vect.fit(X_train)

# 4. transform training data
X_train_dtm = vect.transform(X_train)

# equivalently: combine fit and transform into a single step
# this is faster and what most people would do
X_train_dtm = vect.fit_transform(X_train)

# examine the document-term matrix
X_train_dtm

# 4. transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm

# you can see that the number of columns, 7456, is the same as what we have learned above in X_train_dtm


# 1. import
from sklearn.naive_bayes import MultinomialNB

# 2. instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()
# 3. train the model 

nb.fit(X_train_dtm, y_train)

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
#Naive bayes is fast as seen above
#This matters when we're using 10-fold cross-validation with a large dataset

# 4. make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# examine class distribution
print(y_test.value_counts())
# there is a majority class of 0 here, hence the classes are skewed

# calculate null accuracy (for multi-class classification problems)
# .head(1) assesses the value 1208
null_accuracy = y_test.value_counts().head(1) / len(y_test)
print('Null accuracy:', null_accuracy)

# Manual calculation of null accuracy by always predicting the majority class
print('Manual null accuracy:',(1208 / (1208 + 185)))

#In this case, we can see that our accuracy (0.9885) is higher than the null accuracy (0.8672)

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)

#[TN FP 
#FN TP]
# print message text for the false positives (ham incorrectly classified as spam)

X_test[y_pred_class > y_test]

# alternative less elegant but easier to understand
X_test[(y_pred_class==1) & (y_test==0)]

# print message text for the false negatives (spam incorrectly classified as ham)

X_test[y_pred_class < y_test]
# alternative less elegant but easier to understand
X_test[(y_pred_class==0) & (y_test==1)]

# example false negative
X_test[3132]

# calculate predicted probabilities for X_test_dtm (poorly calibrated)

# Numpy Array with 2C
# left Column: probability class 0
# right C: probability class 1
# we only need the right column 
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob

# Naive Bayes predicts very extreme probabilites, you should not take them at face value

# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)
