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


#df = (sqlc.read.format('com.databricks.spark.csv').options(header='false', inferschema='true').load('data/sonar.all-data.txt'))
df = (sqlc.read.format('com.databricks.spark.csv').options(header='false', inferschema='true').load('C:\\Users\\y9ck3\GITHUB\\Sentiment_Analysis\\train.csv'))

df.printSchema()















'''
df = df.withColumnRenamed("C60","label")

assembler = VectorAssembler(
    inputCols=['C%d' % i for i in range(60)],
    outputCol="features")
output = assembler.transform(df)

standardizer = StandardScaler(withMean=True, withStd=True,
                              inputCol='features',
                              outputCol='std_features')
model = standardizer.fit(output)
output = model.transform(output)

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

'''



'''
#https://www.youtube.com/watch?v=32q7Gn9XjiU

from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import json

sc = SparkContext(appName = "ReutersClassification")
sqlContext = SQLContext(sc)

articles = sc.parallelize(loadData("data/","reuters"))
articles_filtered = articles.filter(lambda article:"topics" in article and "body" in article )
bodies_and_topics = article_filtered.map(lambda article: [article["body"], float("earn" in article["topics"])])
data = bodies_and_topics.toDF(["text"],["label"])

trainset, testset = data.randomSplit([0.8,0.2])

tokenizer = Tokenizer(inputCol="text", outputCol = "words")
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

sc.stop()
'''





'''
#https://github.com/juliensimon/dlnotebooks/blob/master/spark/01%20-%20Spam%20classifier.ipynb
from pyspark import SparkContext, SparkConf
#import sagemaker_pyspark

#conf = (SparkConf().set("spark.driver.extraClassPath", ":".join(sagemaker_pyspark.classpath_jars())))

conf = SparkConf() #.set("spark.driver.extraClassPath", ":".join(sagemaker_pyspark.classpath_jars())))

sc = SparkContext(conf=conf)

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS, SVMWithSGD, NaiveBayes
from pyspark.mllib.tree import DecisionTree, GradientBoostedTrees, RandomForest
from pyspark.mllib.evaluation import MulticlassMetrics

spam = sc.textFile("NDX.csv")
print(type(spam))
#print(spam.count())
'''



'''
#pyspark --packages com.databricks:spark-csv_2.10:1.2.0

from pyspark.sql import SQLContext
from pyspark.sql.types import *
sqlContext = SQLContext(sc)
'''




'''
#https://decisionstats.com/2017/08/31/importing-data-from-csv-file-using-pyspark/
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("Data Cleaning").getOrCreate()

dataframe2 = spark.read.format("csv").option("header","true").option("mode","DROPMALFORMED").load("NDX.csv")

print(type(dataframe2))
'''



'''
from pyspark.ml.classification import LogisticRegression

from pyspark.context import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)


traing = spark.read.csv("NDX.csv", header = True)


lr= LogisticRegression(maxIter=10, regParam=0.3,elasticNetParam=0.8)

lrModel = lr.fit(training)

print("Coefficients:" + str(lrModel.coefficients))

'''




'''
from pyspark.ml.classification import LogisticRegression

# Load training data
training = sc.textFile("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# We can also use the multinomial family for binary classification
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# Fit the model
mlrModel = mlr.fit(training)

# Print the coefficients and intercepts for logistic regression with multinomial family
print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
print("Multinomial intercepts: " + str(mlrModel.interceptVector))

'''

sc.stop()
