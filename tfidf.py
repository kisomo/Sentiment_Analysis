import numpy as np

#http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print(twenty_train.keys())


#['description', 'DESCR', 'filenames', 'target_names', 'data', 'target']

print(twenty_train.target_names)

print(len(twenty_train.data))

print(len(twenty_train.filenames))
print("\n\n")
print(twenty_train.filenames)

print("\n\n")
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print("\n")
print(twenty_train.target_names[twenty_train.target[0]])
print("\n")
print(twenty_train.target[:10])

print("+++++++++++++++++++++++++++++ CounteVectorizer+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

print(count_vect.vocabulary_.get(u'algorithm'))

print("+++++++++++++++++++++++++++++ TF-IDF +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
#To try to predict the outcome on a new document we need to extract the features using almost the same feature 
#extracting chain as before. The difference is that we call transform instead of fit_transform on the 
#transformers, since they have already been fit to the training set:

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
print(predicted)
print("\n\n")
for doc, category in zip(docs_new, predicted):
     print('%r => %s' % (doc, twenty_train.target_names[category]))


#Pipeline ++++++++++++++++++++++++++++

#In order to make the vectorizer => transformer => classifier easier to work with, scikit-learn provides a 
#Pipeline class that behaves like a compound classifier:

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])

#The names vect, tfidf and clf (classifier) are arbitrary. We shall see their use in the section on grid search,
# below. We can now train the model with a single command:

print(text_clf.fit(twenty_train.data, twenty_train.target))  

#Evaluation ++++++++++++++++++++++++++++++++++

#Evaluating the predictive accuracy of the model is equally easy:

import numpy as np
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))            

#SVM ++++++++++++++++++++++++++++++++++++++++++++++++
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', 
penalty='l2', alpha=1e-3, random_state=42,  max_iter=5, tol=None)), ])

print(text_clf.fit(twenty_train.data, twenty_train.target) ) 

predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))            

#scikit-learn further provides utilities for more detailed performance analysis of the results:

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
                                   

print(metrics.confusion_matrix(twenty_test.target, predicted))

#As expected the confusion matrix shows that posts from the newsgroups on atheism and christian 
#are more often confused for one another than with computer graphics.


#PARAMETER SEARCH +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#We have already encountered some parameters such as use_idf in the TfidfTransformer. Classifiers tend to have many parameters 
#as well; e.g., MultinomialNB includes a smoothing parameter alpha and SGDClassifier has a penalty parameter alpha and 
#configurable loss and penalty terms in the objective function (see the module documentation, or use the Python help function,
# to get a description of these).
#Instead of tweaking the parameters of the various components of the chain, it is possible to run an exhaustive search of 
#the best parameters on a grid of possible values. We try out all classifiers on either words or bigrams, with or without idf, 
#and with a penalty parameter of either 0.01 or 0.001 for the linear SVM:

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3), }

#Obviously, such an exhaustive search can be expensive. If we have multiple CPU cores at our disposal, we can tell the grid 
#searcher to try these eight parameter combinations in parallel with the n_jobs parameter. If we give this parameter 
#a value of -1, grid search will detect how many cores are installed and uses them all:

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
#The grid search instance behaves like a normal scikit-learn model. Let us perform the search on a smaller subset of the 
#training data to speed up the computation:

gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
#The result of calling fit on a GridSearchCV object is a classifier that we can use to predict:

print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])

#The object s best_score_ and best_params_ attributes store the best mean score and the parameters setting corresponding 
#to that score:

print(gs_clf.best_score_)                   

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


#A more detailed summary of the search is available at gs_clf.cv_results_.
#The cv_results_ parameter can be easily imported into pandas as a DataFrame for further inspection.

print("++++++++++++++++++++++++++++++ Exercises +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")






