# -*- coding: utf-8 -*-

'''
#https://towardsdatascience.com/sentiment-analysis-with-pyspark-bc8e83f80c35


import numpy as np

#import findspark
#findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext


try:
    # create SparkContext on all CPUs available: in my case I have 4 CPUs on my laptop
    sc = ps.SparkContext('local[8]')
    sqlContext = SQLContext(sc)
    print("Just created a SparkContext")
except ValueError:
    warnings.warn("SparkContext already exists in this scope")

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', 
inferschema='true').load('/home/terrence/CODING/Python/MODELS/Reviews.csv')

type(df)


df = df.dropna()
df.count()

#raw_df = pd.read_csv("/home/terrence/CODING/Python/MODELS/Reviews.csv")


from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("Linear Regression Model").config("spark.executor.memory","1gb").getOrCreate()

sc = spark.sparkContext


train = sc.textFile("train.csv")
est = sc.textFile("test.csv")
subm = sc.textFile("sample_submission.csv")

print(type(train))
print(type(test))
'''

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
#https://github.com/juliensimon/dlnotebooks/blob/master/spark/01%20-%20Spam%20classifier.ipynb

from pyspark import SparkContext, SparkConf
#import sagemaker_pyspark

conf = SparkConf() #.set("spark.driver.extraClassPath", ":".join(sagemaker_pyspark.classpath_jars())))
sc = SparkContext(conf=conf)

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS, SVMWithSGD, NaiveBayes
from pyspark.mllib.tree import DecisionTree, GradientBoostedTrees, RandomForest
from pyspark.mllib.evaluation import MulticlassMetrics

train = pd.read_csv("train.csv")
print(train.shape)

sc.stop()
'''

'''
print("+++++++++++++++++++++++++++++++++++ sklearn ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
#print(os.listdir("../input"))

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#train = pd.read_csv('C:\\Users\\y9ck3\\GITHUB\\Sentiment_Analysis\\train.csv')
train = pd.read_csv('train.csv')

#test = pd.read_csv('C:\\Users\\y9ck3\\GITHUB\\Sentiment_Analysis\\test.csv')
test = pd.read_csv('test.csv')

#subm1 = pd.read_csv('sample_submission.csv')
subm2 = pd.read_csv('sample_submission.csv')
#subm3 = pd.read_csv('sample_submission.csv')
#subm4 = pd.read_csv('sample_submission.csv')
#subm5 = pd.read_csv('sample_submission.csv')
#subm6 = pd.read_csv('C:\\Users\y9ck3\\GITHUB\\Sentiment_Analysis\\sample_submission.csv')
subm6 = pd.read_csv('sample_submission.csv')

#subm5 = pd.read_csv('../input/sample_submission.csv')

print(train.shape)
print(test.shape)
print(train.head(3))
print(test.head(3))
print(train['comment_text'][0])

lens = train.comment_text.str.len()
lens.mean(), lens.std(), lens.max()


lens.hist()

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train[label_cols].max(axis=1)
train['none'] = 1-train[label_cols].max(axis=1)


COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

import re
import string
re_tok = re.compile(r'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r'\1', s).split()

print("printing classes\n\n")
print(label_cols[0])
print(label_cols[1])
print(label_cols[2])
print(label_cols[3])
print(label_cols[4])
print(label_cols[5])
#print(label_cols[6])


n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

preds1 = np.zeros((len(test), len(label_cols)))
preds2 = np.zeros((len(test), len(label_cols)))
preds3 = np.zeros((len(test), len(label_cols)))
preds4 = np.zeros((len(test), len(label_cols)))
preds5 = np.zeros((len(test), len(label_cols)))
preds6 = np.zeros((len(test), len(label_cols)))
'''


'''
trn_term_doc.to_csv('X.csv', index=False, float_format = "%.8f")

train[0].to_csv('y0.csv', index=False, float_format = "%.8f")
train[1].to_csv('y1.csv', index=False, float_format = "%.8f")
train[2].to_csv('y2.csv', index=False, float_format = "%.8f")
train[3].to_csv('y3.csv', index=False, float_format = "%.8f")
train[4].to_csv('y4.csv', index=False, float_format = "%.8f")
train[5].to_csv('y5.csv', index=False, float_format = "%.8f")



_length = trn_term_doc.shape[0]
_width = len(label_cols)+1
print(_length)
print(_width)


df_0 = np.zeros((_length, _width))
df_0 = pd.DataFrame(df_0)
trn = pd.DataFrame(trn_term_doc)
trn0 = pd.DataFrame(train[0])

df_0 = pd.concat([trn, trn0], axis =1)
#df_0[:,:-1] = np.array(trn_term_doc)
#df_0[:,-1] = train[0]
#print(df_0[:3,:])

print(df_0.shape)
print(df_0.head(2))

df_0.to_csv('mission_0.csv', index=False, float_format = "%.8f")
'''

'''
df_2 = pd.concat([trn_term_doc, train[2]], axis=1)
df_2.to_csv('mission_2.csv', index=False, float_format="%.8f")

df_3 = pd.concat([trn_term_doc, train[3]], axis=1)
df_3.to_csv('mission_3.csv', index=False, float_format="%.8f")

df_4 = pd.concat([trn_term_doc, train[4]], axis=1)
df_4.to_csv('mission_4.csv', index=False, float_format="%.8f")

df_5 = pd.concat([trn_term_doc, train[5]], axis=1)
df_5.to_csv('mission_5.csv', index=False, float_format="%.8f")

df_6 = pd.concat([trn_term_doc, train[6]], axis=1)
df_6.to_csv('mission_6.csv', index=False, float_format="%.8f")

'''

'''
for i, j in enumerate(label_cols):
    print('fit', j)
    df_j = pd.concat([trn_term_doc, train[j]], axis=1)
    df_j.to_csv('mission_j.csv', index=False, float_format="%.8f")
    

#submid1 = pd.DataFrame({'id': subm1["id"]})
#submission_m2 = pd.concat([submid1, pd.DataFrame(preds1, columns = label_cols)], axis=1)
#submission_m2.to_csv('submission_m2.csv', index=False, float_format="%.8f")
'''



'''
nb = MultinomialNB(alpha=1, class_prior=None, fit_prior=True)
for i, j in enumerate(label_cols):
    print('fit', j)
    #m,r = get_mdl(train[j])
    m= nb.fit(trn_term_doc, train[j])
    #preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    preds1[:,i] = m.predict_proba(test_term_doc)[:,1]
    #preds1[:,i] = m.predict(test_term_doc)


print(preds1[:3,:])


submid1 = pd.DataFrame({'id': subm1["id"]})
submission_m2 = pd.concat([submid1, pd.DataFrame(preds1, columns = label_cols)], axis=1)
submission_m2.to_csv('submission_m2.csv', index=False, float_format="%.8f")
'''

'''
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
#scaler = MinMaxScaler()
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(trn_term_doc)

#X_test_scaled = scaler.transform(X_test)

#from sklearn.linear_model import SGDClassifier
#lr = SGDClassifier(loss='log',penalty='elasticnet',alpha = 0.5, l1_ratio = 0.7, max_iter=500, n_jobs =-1)

lr = LogisticRegression(penalty = 'l1',solver = 'saga', max_iter = 1000, n_jobs = -1)
for i, j in enumerate(label_cols):
    print('fit', j)
    m_lr= lr.fit(X_train_scaled, train[j])
    preds2[:,i] = m_lr.predict_proba(scaler.transform(test_term_doc))[:,1]


print(preds2[:3,:])

#subm2 = pd.read_csv('sample_submission.csv')
submid2 = pd.DataFrame({'id': subm2["id"]})
submission_lr = pd.concat([submid2, pd.DataFrame(preds2, columns = label_cols)], axis=1)
submission_lr.to_csv('submission_lr.csv', index=False, float_format="%.8f")
'''


'''

clf = GradientBoostingClassifier()
for i, j in enumerate(label_cols):
    print('fit', j)
    #m,r = get_mdl(train[j])
    m_clf= clf.fit(trn_term_doc, train[j])
    #preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    preds3[:,i] = m_clf.predict_proba(test_term_doc)[:,1]
    #preds3[:,i] = m_clf.predict(test_term_doc)
'''

'''
submid3 = pd.DataFrame({'id': subm3["id"]})
submission_g = pd.concat([submid3, pd.DataFrame(preds3, columns = label_cols)], axis=1)
submission_g.to_csv('submission_g.csv', index=False, float_format="%.8f")
'''

'''
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', C = 1, gamma = 1)
for i, j in enumerate(label_cols):
    print('fit', j)
    #m,r = get_mdl(train[j])
    m_svc= svc.fit(trn_term_doc, train[j])
    #preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    preds4[:,i] = m_svc.predict_proba(test_term_doc)[:,1]
    #preds4[:,i] = m_svc.predict(test_term_doc)
    
print(preds4[:3,:])

submid4 = pd.DataFrame({'id': subm4["id"]})
submission_sv = pd.concat([submid4, pd.DataFrame(preds4, columns = label_cols)], axis=1)
submission_sv.to_csv('submission_sv.csv', index=False, float_format="%.8f")
'''



'''
from sklearn.neighbors.nearest_centroid import NearestCentroid
clf_nc = NearestCentroid()

for i, j in enumerate(label_cols):
    print('fit', j)
    #m,r = get_mdl(train[j])
    m_nc= clf_nc.fit(trn_term_doc, train[j])
    #preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    preds4[:,i] = m_nc.predict_proba(test_term_doc)[:,1]
    #preds4[:,i] = m_svc.predict(test_term_doc)


submid4 = pd.DataFrame({'id': subm4["id"]})
submission_nc = pd.concat([submid4, pd.DataFrame(preds4, columns = label_cols)], axis=1)
submission_nc.to_csv('submission_nc.csv', index=False, float_format="%.8f")

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=6)
#for i, j in enumerate(label_cols):
    #print('fit', j)
    #m,r = get_mdl(train[j])
m_gmm= gmm.fit(trn_term_doc) #, train[j])
    #preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
preds4[:,i] = m_gmm.predict_proba(test_term_doc)[:,1]
    #preds4[:,i] = m_svc.predict(test_term_doc)
'''


'''
#preds_3_1 = np.average((preds1,preds2,preds3), axis = 0)


#print(preds3.shape)
#print(preds_3_1.shape)

#submid_3_1 = pd.DataFrame({'id': subm5["id"]})
#submission_3_1 = pd.concat([submid_3_1, pd.DataFrame(preds_3_1, columns = label_cols)], axis=1)
#submission_3_1.to_csv('submission_3_1.csv', index=False, float_format="%.8f")


clf_nn = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(100,100,100,100),random_state=1)
for i, j in enumerate(label_cols):
    print('fit', j)
    m_clf_nn= clf_nn.fit(trn_term_doc, train[j])
    preds6[:,i] = m_clf_nn.predict_proba(test_term_doc)[:,1]
    #preds4[:,i] = m_clf_nn.predict(test_term_doc)


submid6 = pd.DataFrame({'id': subm6["id"]})
submission_nn = pd.concat([submid6, pd.DataFrame(preds6, columns = label_cols)], axis=1)
submission_nn.to_csv('C:\\Users\\y9ck3\\GITHUB\\Sentiment_Analysis\\submission_nn.csv', index=False, float_format="%.8f")

#preds5 = np.average((preds1,preds2,preds3,preds4), axis = 0)


#submid5 = pd.DataFrame({'id': subm5["id"]})
#submission_en = pd.concat([submid5, pd.DataFrame(preds5, columns = label_cols)], axis=1)
#submission_en.to_csv('submission_en.csv', index=False, float_format="%.8f")
'''




print("+++++++++++++++++++++++++++++++++++++++++++ spark +++++++++++++++++++++++++++++++++++++++++++++++++++++++")

#http://people.duke.edu/~ccc14/sta-663-2016/21D_Spark_MLib.html

from pyspark import SparkContext
sc = SparkContext('local[*]')

from pyspark.sql import SQLContext
sqlc = SQLContext(sc)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import PCA
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import GaussianMixture
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

'''
#df = (sqlc.read.format('com.databricks.spark.csv').options(header='false', inferschema='true').load('data/sonar.all-data.txt'))
df = (sqlc.read.format('com.databricks.spark.csv').options(header='false', inferschema='true').load('C:\\Users\\y9ck3\GITHUB\\Sentiment_Analysis\\train.csv'))

df.printSchema()

#df.select("_c0").show()

df = df.withColumnRenamed("C0","comment_text")
df = df.withColumnRenamed("C1","label")

#assembler = VectorAssembler(
#    inputCols=['C%d' % i for i in range(60)],
#    outputCol="features")
#output = assembler.transform(df)

#standardizer = StandardScaler(withMean=True, withStd=True,
#                              inputCol='features',
#                              outputCol='std_features')
#model = standardizer.fit(output)
#output = model.transform(output)


#df = df.withColumnRenamed("C0","comment_text")
df = df.withColumnRenamed("C1","label")
assembler = VectorAssembler(inputCols="comment_text", outputCol="features")
output = assembler.transform(df)


from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.evaluation import BinaryClassificationEvaluator



#articles = sc.parallelize(loadData("data/","reuters"))
#articles_filtered = articles.filter(lambda article:"topics" in article and "body" in article )
#bodies_and_topics = article_filtered.map(lambda article: [article["body"], float("earn" in article["topics"])])
#data = bodies_and_topics.toDF(["text"],["label"])

data = df.select("_c0", "_c1")
data = data.toDF(["text"],["label"])
#data = df.select("comment_text", "label")

trainset, testset = data.randomSplit([0.8,0.2])

#tokenizer = Tokenizer(inputCol="text", outputCol = "words")
tokenizer = Tokenizer(inputCol="_c0", outputCol = "words")
hasher = HashingTF(inputCol = tokenizer.getOutputCol(),outputCol = "features")
clf = LogisticRegression(maxIter = 10, regParam = 0.01)
pipeline = Pipeline(stages = [tokenizer, hasher, clf])

model = pipeline.fit(trainset)

train_pred = model.transform(trainset)
test_pred = model.transform(testset)

evaluator = BinaryClassificationEvaluator()

train_accuracy = evaluator.evaluate(train_pred)
print("Train set accuracy: {:.3g}".format(train_accuracy))
test_accuracy = evaluator.evaluate(test_pred)
print("Test set accuracy: {:.3g}".format(test_accuracy))











indexer = StringIndexer(inputCol="label", outputCol="label_idx")
indexed = indexer.fit(output).transform(output)

sonar = indexed.select(['std_features', 'label', 'label_idx'])

sonar.show(n=3)

pca = PCA(k=2, inputCol="std_features", outputCol="pca")
model = pca.fit(sonar)
transformed = model.transform(sonar)

features = transformed.select('pca').rdd.map(lambda x: np.array(x))

gmm = GaussianMixture.train(features, k=2)

predict = gmm.predict(features).collect()

labels = sonar.select('label_idx').rdd.map(lambda r: r[0]).collect()

np.corrcoef(predict, labels)

xs = np.array(features.collect()).squeeze()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(xs[:, 0], xs[:,1], c=predict)
axes[0].set_title('Predicted')
axes[1].scatter(xs[:, 0], xs[:,1], c=labels)
axes[1].set_title('Labels')
pass

sonar.show(n=3)

data = sonar.map(lambda x: LabeledPoint(x[2], x[0]))

train, test = data.randomSplit([0.7, 0.3])

model = LogisticRegressionWithLBFGS.train(train)

y_yhat = test.map(lambda x: (x.label, model.predict(x.features)))
err = y_yhat.filter(lambda x: x[0] != x[1]).count() / float(test.count())
print("Error = " + str(err))

# using ml

transformer = VectorAssembler(inputCols=['C%d' % i for i in range(60)],
                              outputCol="features")
standardizer = StandardScaler(withMean=True, withStd=True,
                              inputCol='features',
                              outputCol='std_features')
indexer = StringIndexer(inputCol="C60", outputCol="label_idx")
pca = PCA(k=5, inputCol="std_features", outputCol="pca")
lr = LogisticRegression(featuresCol='std_features', labelCol='label_idx')

pipeline = Pipeline(stages=[transformer, standardizer, indexer, pca, lr])

df = (sqlc.read.format('com.databricks.spark.csv')
      .options(header='false', inferschema='true')
      .load('data/sonar.all-data.txt'))


train, test = df.randomSplit([0.7, 0.3])

model = pipeline.fit(train)

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    prediction = model.transform(test)


score = prediction.select(['label_idx', 'prediction'])
score.show(n=score.count())

acc = score.map(lambda x: x[0] == x[1]).sum() / score.count()
acc

#pip install spark-sklearn

from sklearn import svm, grid_search, datasets
from spark_sklearn import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = GridSearchCV(sc, svr, parameters)
clf.fit(iris.data, iris.target)

GridSearchCV(cv=None, error_score='raise',
       estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'kernel': ('linear', 'rbf'), 'C': [1, 10]},
       pre_dispatch='2*n_jobs', refit=True,
       sc=<pyspark.context.SparkContext object at 0x11ad38668>,
       scoring=None, verbose=0)




sc.stop()
'''









