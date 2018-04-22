import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Embeddings
#https://embeddings.macheads101.com/


##fake news datacamp
df = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv")
print(df.shape)
print(df.head(3))
##df.to_csv('fake_or_real_news.csv') 

'''
df = pd.read_csv("fake_or_real_news.csv")

print(df.shape)
print(df.head(3))

print("\n\n")
print(df.columns.values)

df1 = df['Unnamed: 0.1', 'title', 'text', 'label']

print(df1.shape)
print(df1.head(3))

print("\n\n")
print(df1.columns.values)
'''

df = df.set_index("Unnamed: 0")
print(df.head(3))

y = df.label
df.drop("label", axis =1)

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test, y_train, y_test = train_test_split(df['text'],y, test_size=0.33, random_state=53)


count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.fit_transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english',max_df=0.7)
#tfidf_vectorizer = TfidfTransformer(stop_words = 'english',max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.fit_transform(X_test)


print(tfidf_vectorizer.get_feature_names()[-10:])

print(count_vectorizer.get_feature_names()[:10])


# done with data preparation now create the confusion matrix function

def plot_confusion_matrix(cm, classes, normalizer=False, title = 'Confusion Matrix', cmap =plt.cm.Blues):
    
    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arrange(len(classes))
    plt.xticks(tick_marks, classes, rotation =45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    else:
        print('Confusion matrix, Without normalization')
    
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j], horizontalalignment='center',color='white' if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#Multinomial Naive Bayes with countvectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf = MultinomialNB()

clf.fit(tfidf_train,y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test,pred)
print("accuracy:  %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
plot_confusion_matrix(cm, classes = ['FAKE','REAL'])

























#word2vec
#http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/


















#organizing emails
#http://www.andreykurenkov.com/writing/ai/organizing-my-emails-with-a-neural-net/



















#Toxic comments
#https://www.kaggle.com/emannueloc/using-word-embeddings-with-gensim















#pre-trained glove
#https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python















#glove bots
#https://medium.com/rasa-blog/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d

















#pre-trained office automation
#https://www.youtube.com/watch?v=hEE8ZxcXxu4&feature=youtu.be







