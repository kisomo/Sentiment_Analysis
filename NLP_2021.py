# -*- coding: utf-8 -*-

#sudo docker build . -t sentiment:1.0.0
#sudo docker images
#sudo docker run -d -p 5000:5000 sentiment:1.0.0
#sudo docker run --rm -it sentiment:1.0.0

#docker system prune -a

from tabulate import tabulate
#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from sklearn.cross_validation import cross_val_score
#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split


################################ SPAM CLASSIFIER  ################################################################################################################

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
#from sklearn.cross_validation import train_test_split
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
X_train = X_train.toarray()
X_train_dtm = transformer.fit_transform(X_train)

# examine the document-term matrix
#X_train_dtm
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

#https://www.kaggle.com/selener/multi-class-text-classification-tfidf

'''
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')

print(sms.head())

# We transform each complaint into a vector
features = tfidf.fit_transform(sms.message).toarray()

labels = sms.label_num

print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))


X = sms['message'] # Collection of documents
y = sms['label'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
print(acc)


plt.figure(figsize=(8,5))

sns.boxplot(x='model_name', y='accuracy',  data=cv_df, color='lightblue', showmeans=True)

plt.title("MEAN ACCURACY (cv = 5)\n", size=14)
plt.show()

#Model Evaluation

X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, labels, sms.index, test_size=0.25, random_state=1)
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification report
print('\t\t\t\tCLASSIFICATIION METRICS\n')
print(metrics.classification_report(y_test, y_pred, target_names= sms['label'].unique()))


# Create a new column 'category_id' with encoded categories 
sms['category_id'] = sms['label'].factorize()[0]
category_id_df = sms[['label', 'category_id']].drop_duplicates()


# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label']].values)

# New dataframe
sms.head()


conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=category_id_df.label.values, 
            yticklabels=category_id_df.label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=16)


for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 20:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], 
                                                           id_to_category[predicted], 
                                                           conf_mat[actual, predicted]))
    
      display(df2.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['label', 
                                                                'message']])
      print('')


#Most correlated terms with each category

model.fit(features, labels)

N = 4
for Product, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("\n==> '{}':".format(Product))
  print("  * Top unigrams: %s" %(', '.join(unigrams)))
  print("  * Top bigrams: %s" %(', '.join(bigrams)))


#Now let's make a few predictions on unseen data.

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')

fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

model = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)


new_complaint = """I have been enrolled back at XXXX XXXX University in the XX/XX/XXXX. Recently, i have been harassed by \
Navient for the last month. I have faxed in paperwork providing them with everything they needed. And yet I am still getting \
phone calls for payments. Furthermore, Navient is now reporting to the credit bureaus that I am late. At this point, \
Navient needs to get their act together to avoid me taking further action. I have been enrolled the entire time and my \
deferment should be valid with my planned graduation date being the XX/XX/XXXX."""

print(model.predict(fitted_vectorizer.transform([new_complaint])))

sms[sms['message'] == new_complaint]


new_complaint_2 = """Equifax exposed my personal information without my consent, as part of their recent data breach. \
In addition, they dragged their feet in the announcement of the report, and even allowed their upper management to sell \
off stock before the announcement."""

print(model.predict(fitted_vectorizer.transform([new_complaint_2])))

sms[sms['message'] == new_complaint_2]

'''



print("++++++++++++++++++++++++++++++++++++++++++++ word2vec ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

''''
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
print (model.most_similar('education'))
print("\n")



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


#----------------------------- TF-IDF ------------------------------

# ---------------------------- Word2vec ----------------------------

# --------------------------- ELMO --------------------------------

'''

#https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/

import spacy
from tqdm import tqdm
import re
import time
import pickle
pd.set_option('display.max_colwidth', 200)

# check that the conversion worked
print(sms.head())

train, test = train_test_split(sms, test_size=0.25, random_state = 0)

print(train.shape)
print(test.shape)

print(train['label'].value_counts(normalize = True))

print(train.head())

#Text Cleaning and Preprocessing


#remove emojis
train['clean_message'] = train['message'].astype(str).str.encode('ascii','ignore').str.decode('ascii')
test['clean_message'] = test['message'].astype(str).str.encode('ascii','ignore').str.decode('ascii')


# remove URL's from train and test
train['clean_message'] = train['message'].apply(lambda x: re.sub(r'http\S+', '', x))
test['clean_message'] = test['message'].apply(lambda x: re.sub(r'http\S+', '', x))


# remove punctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

train['clean_message'] = train['clean_message'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
test['clean_message'] = test['clean_message'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

# convert text to lowercase
train['clean_message'] = train['clean_message'].str.lower()
test['clean_message'] = test['clean_message'].str.lower()

# remove numbers
train['clean_message'] = train['clean_message'].str.replace("[0-9]", " ")
test['clean_message'] = test['clean_message'].str.replace("[0-9]", " ")

# remove whitespaces
train['clean_message'] = train['clean_message'].apply(lambda x:' '.join(x.split()))
test['clean_message'] = test['clean_message'].apply(lambda x: ' '.join(x.split()))


# import spaCy's language model
#nlp = spacy.load('en', disable=['parser', 'ner'])
nlp = spacy.load("en_core_web_sm")

# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

train['clean_message'] = lemmatization(train['clean_message'])
test['clean_message'] = lemmatization(test['clean_message'])


print(train.sample(10))

#$ pip install "tensorflow>=1.7.0"
#$ pip install tensorflow-hub

import tensorflow_hub as hub
import tensorflow as tf

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

#elmo_model = hub.load("https://tfhub.dev/google/elmo/2")
#out = elmo_model.signatures["default"](in)

# just a random sentence
x = ["Roasted ants are a popular snack in Columbia"]
# Extract ELMo features 
embeddings = elmo(x, signature="default", as_dict=True)["elmo"]
print(embeddings.shape)

'''



#--------------------------- BERT ---------------------------------

'''
import torch

print(train.shape)
print(test.shape)
print("------------------------------------------------------------")
print(train.head())
print(test.head())

#import the transformers model
from simpletransformers.classification import ClassificationModel

#ensure that you can use cuda if it is available, but won't error out if it is not.
cuda_available = torch.cuda.is_available()

# define hyperparameters
train_args ={"reprocess_input_data": True,
             "fp16":False,
             "use_cuda":cuda_available,
             "train_batch_size":8,
             "num_train_epochs": 5}

# Create a ClassificationModel
# we use the "bert-base-multilingual-uncased" model which is trained on top 102 wikipedia languages including Urdu
model1 = ClassificationModel(
    "bert", "bert-base-multilingual-cased",
    num_labels=3,
    use_cuda = cuda_available,
    args=train_args
)

#train the model - this will take sometime
# if you have powerful runtime you could set more epochs from the args above
model1.train_model(train, output_dir = "/model1")


#create a helper function for calculating the performance metrics
from sklearn.metrics import f1_score, accuracy_score
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


#calculate the performance metrics
result, model_outputs, wrong_predictions = model1.eval_model(test, f1=f1_multiclass, acc=accuracy_score)

result

model_outputs

#extract the true text content in the test data set as a list
y_test = test.label
x_test = []
for txt in test.text:
    tok = txt
    x_test.append(tok)

#compute the predicted values
y_pred = model1.predict(x_test)

y_pred = y_pred[0]

test_acc = accuracy_score(y_test,y_pred)
test_acc 

#class names
class_names = ['Positive', 'Neutral', 'Negative']

#define the confusion matrix
def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment')


cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

#use the model to predict the sentiment of a raw social media post
model1.predict(["are wha kaya bat hai"])

#Remember how we encoded our labels and use that encoding to reveal the true sentiment
# 1 = Neutral
#0 = Positive
#2 = Negative

class_list = ['Positive', 'Neutral', 'Negative']

text1 = "are wha kaya bat hai"

predicts, raw_outputs = model1.predict([text1])

#Show the real sentiment
print(class_list[predicts[0]])
'''


#---------------------------- GPT2 -------------------------------

'''

#https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/

print(sms.shape)
print(sms.head())

import io
import os
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
#from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

# Set seed for reproducibility.
set_seed(123)

# Number of training epochs (authors on fine-tuning Bert recommend between 2 and 4).
epochs = 4

# Number of batches - depending on the max sequence length and GPU memory.
# For 512 sequence length batch of 10 works without cuda memory issues.
# For small sequence length can try batch of 32 or higher.
batch_size = 32

# Pad or truncate text sequences to a specific length
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = 60

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Name of transformers model - will use already pretrained model.
# Path of transformer model - will load your own model from local disk.
model_name_or_path = 'gpt2'

# Dictionary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids = {'neg': 0, 'pos': 1}

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

'''






#---------------------------- GPT3 ------------------------------





################################ TOXIC COMMENTS ################################################################################################################

#----------------------------- TF-IDF ------------------------------

# ---------------------------- Word2vec ----------------------------

# --------------------------- ELMO --------------------------------

#--------------------------- BERT ---------------------------------

#---------------------------- GPT2 -------------------------------

#---------------------------- GPT3 ------------------------------









