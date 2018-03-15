
#https://towardsdatascience.com/sentiment-analysis-with-pyspark-bc8e83f80c35

import numpy as np

#import findspark
#findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext

'''
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
'''
#raw_df = pd.read_csv("/home/terrence/CODING/Python/MODELS/Reviews.csv")

from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("Linear Regression Model").config("spark.executor.memory","1gb").getOrCreate()

sc = spark.sparkContext


train = sc.textFile("train.csv")
test = sc.textFile("test.csv")
subm = sc.textFile("sample_submission.csv")



