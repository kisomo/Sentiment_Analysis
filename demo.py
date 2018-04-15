import numpy as np
import pandas as pd

#http://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/

#https://appliedmachinelearning.wordpress.com/2017/02/12/sentiment-analysis-using-tf-idf-weighting-pythonscikit-learn/



# alternative: read file into pandas from a URL
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_table(url, header=None, names=['label', 'message'])

# examine the shape
print(sms.shape)
print(sms.head(3)) #TM

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

print("+++++++++++++++++++++++++++++++++ CountVectorizer ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# 1. import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer

# 2. instantiate CountVectorizer (vectorizer)
vect = CountVectorizer()
print(vect) #TM

# learn training data vocabulary, then use it to create a document-term matrix

# 3. fit
vect.fit(X_train)
print('\n\n')
print(vect) #TM

# 4. transform training data
#X_train_dtm = vect.transform(X_train)

# equivalently: combine fit and transform into a single step
# this is faster and what most people would do
X_train_dtm = vect.fit_transform(X_train)

# examine the document-term matrix
X_train_dtm
print(X_train_dtm)

all_words = vect.get_feature_names() #TM
print(all_words) #TM

# 4. transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm
print(X_test_dtm.toarray())
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
print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.precision_score(y_test, y_pred_class))
print(metrics.recall_score(y_test, y_pred_class))

#sent1 = np.array(['I am not happy']) #TM
#print(nb.predict(sent1.reshape(1,-1))) #TM


# examine class distribution
print(y_test.value_counts())
# there is a majority class of 0 here, hence the classes are skewed

# calculate null accuracy (for multi-class classification problems)
# .head(1) assesses the value 1208
null_accuracy = y_test.value_counts().head(1) / len(y_test)
print('\n\nNull accuracy:', null_accuracy)

# Manual calculation of null accuracy by always predicting the majority class
print('Manual null accuracy:',(1208 / (1208 + 185)))

#In this case, we can see that our accuracy (0.9885) is higher than the null accuracy (0.8672)

# print the confusion matrix
print(metrics.confusion_matrix(y_test, y_pred_class))


#[TN FP 
#FN TP]
# print message text for the false positives (ham incorrectly classified as spam)

print(X_test[y_pred_class > y_test])

# alternative less elegant but easier to understand
print(X_test[(y_pred_class==1) & (y_test==0)])

print("\n\n")
# print message text for the false negatives (spam incorrectly classified as ham)

print(X_test[y_pred_class < y_test])
print("\n\n")
# alternative less elegant but easier to understand
print(X_test[(y_pred_class==0) & (y_test==1)])

# example false negative
print(X_test[3132])

# calculate predicted probabilities for X_test_dtm (poorly calibrated)

# Numpy Array with 2C
# left Column: probability class 0
# right C: probability class 1
# we only need the right column 
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
print(y_pred_prob)


# Naive Bayes predicts very extreme probabilites, you should not take them at face value

# calculate AUC
print(metrics.roc_auc_score(y_test, y_pred_prob))


print("+++++++++++++++++++++++++++++++++++++++++++++ tf-idf +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
'''
from sklearn.feature_extraction.text import TfidfTransformer
# create tf-idf object
transformer = TfidfTransformer(smooth_idf=True)
# X can be obtained as X.toarray() from the previous snippet

# learn the vocabulary and store tf-idf sparse matrix in tfidf

# this is faster and what most people would do
X_train.toarray()
X_train_dtm = transformer.fit_transform(X_train)

# examine the document-term matrix
X_train_dtm
print(X_train_dtm)


all_words = vect.get_feature_names() #TM
print(all_words) #TM

# 4. transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm
print(X_test_dtm.toarray())
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
print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.precision_score(y_test, y_pred_class))
print(metrics.recall_score(y_test, y_pred_class))

#sent1 = np.array(['I am not happy']) #TM
#print(nb.predict(sent1.reshape(1,-1))) #TM


# examine class distribution
print(y_test.value_counts())
# there is a majority class of 0 here, hence the classes are skewed

# calculate null accuracy (for multi-class classification problems)
# .head(1) assesses the value 1208
null_accuracy = y_test.value_counts().head(1) / len(y_test)
print('\n\nNull accuracy:', null_accuracy)

# Manual calculation of null accuracy by always predicting the majority class
print('Manual null accuracy:',(1208 / (1208 + 185)))

#In this case, we can see that our accuracy (0.9885) is higher than the null accuracy (0.8672)

# print the confusion matrix
print(metrics.confusion_matrix(y_test, y_pred_class))


#[TN FP 
#FN TP]
# print message text for the false positives (ham incorrectly classified as spam)

print(X_test[y_pred_class > y_test])

# alternative less elegant but easier to understand
print(X_test[(y_pred_class==1) & (y_test==0)])

print("\n\n")
# print message text for the false negatives (spam incorrectly classified as ham)

print(X_test[y_pred_class < y_test])
print("\n\n")
# alternative less elegant but easier to understand
print(X_test[(y_pred_class==0) & (y_test==1)])

# example false negative
print(X_test[3132])

# calculate predicted probabilities for X_test_dtm (poorly calibrated)

# Numpy Array with 2C
# left Column: probability class 0
# right C: probability class 1
# we only need the right column 
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
print(y_pred_prob)


# Naive Bayes predicts very extreme probabilites, you should not take them at face value

# calculate AUC
print(metrics.roc_auc_score(y_test, y_pred_prob))

'''
print("++++++++++++++++++++++++++++++++++++++++++++ word2vec ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
from gensim.models import word2vec
#print(X)
from nltk.corpus import stopwords

#tokenized_sentences = [sentence.split() for sentence in X]
tokenized_sentences = [sentence.split() for sentence in X if not sentence in stopwords.words('english')]
print("\n")
#print(tokenized_sentences)

model = word2vec.Word2Vec(sentences=tokenized_sentences, # tokenized senteces, list of list of strings
                 size=100,  # size of embedding vectors
                 workers=8, # how many threads?
                 min_count=1, # minimum frequency per token, filtering rare words
                 sample=0.05, # weight of downsampling common words
                 sg = 1, # should we use skip-gram? if 0, then cbow
                 iter=10,
                 hs = 0
        )

X = model[model.wv.vocab]

print(X.shape)
print("\n")
print (model.most_similar('clothes'))
print("\n")
'''
print (model.most_similar('open'))
print("\n")
print (model.most_similar('book'))
print("\n")
print(model.most_similar("pick"))
print("\n")
print(model.most_similar("purse"))
#print (model.most_similar(['snack', 'protein'], negative=['supplement']))

'''


'''
#++++++++++++++++++++++++++++++++++ spark +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#nb.fit(X_train_dtm, y_train)
#y_pred_class = nb.predict(X_test_dtm)


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(X_train_dtm)
predictions = lrModel.transform(X_test_dm)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print(evaluator.evaluate(predictions))
'''