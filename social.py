import pandas as pd
import numpy as np


#https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/

import tweepy
from tweepy import OAuthHandler
 
consumer_key = 'YOUR-CONSUMER-KEY'
consumer_secret = 'YOUR-CONSUMER-SECRET'
access_token = 'YOUR-ACCESS-TOKEN'
access_secret = 'YOUR-ACCESS-SECRET'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    print(status.text)

for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    process_or_store(status._json)

for friend in tweepy.Cursor(api.friends).items():
    process_or_store(friend._json)

for tweet in tweepy.Cursor(api.user_timeline).items():
    process_or_store(tweet._json)


def process_or_store(tweet):
    print(json.dumps(tweet))

from tweepy import Stream
from tweepy.streaming import StreamListener
 
class MyListener(StreamListener):
 
    def on_data(self, data):
        try:
            with open('python.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True
 
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#python'])




#https://medium.com/@koshut.takatsuji/twitter-sentiment-analysis-with-full-code-and-explanation-naive-bayes-a380b38f036b

#name = twitter_sentiment_tacosushi
#Import the necessary methods from tweepy library as well as json and time
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import json
import time
import pandas as pd


#We authenticate ourselves as having a twitter app
#Variables that contains the user credentials to access Twitter API 
access_token = "3ads;fkajsdfpoaisdjf;alksdjf"
access_secret = "2uasdl;fajsd;flkjasd;flkajsdf;adfasdfEg"
consumer_key = "xasdf98uoif;wqjer;kandsf"
consumer_secret = "1asd9fp8uijl;qkwef;alksd;iX"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)




#We begin searching our query
#Put your search term
searchquery = "angry"

users =tweepy.Cursor(api.search,q=searchquery).items()
count = 0
start = 0
errorCount=0

#We will be storing our data in file called: happy.json
#file = open('test.json', 'wb') 

#here we tell the program how fast to search 
waitquery = 100      #this is the number of searches it will do before resting
waittime = 2.0          # this is the length of time we tell our program to rest
total_number = 15000     #this is the total number of queries we want
justincase = 1         #this is the number of minutes to wait just in case twitter throttles us



text = [0] * total_number
secondcount = 0
idvalues = [1] * total_number
 #1 is happy; 2 is sad; 3 is angry; 4 is fearful
#Below is where the magic happens and the queries are being made according to our desires above
while secondcount < total_number:
    try:
        user = next(users)
        count += 1
        
        #We say that after every 100 searches wait 5 seconds
        if (count%waitquery == 0):
            time.sleep(waittime)
            #break

    except tweepy.TweepError:
        #catches TweepError when rate limiting occurs, sleeps, then restarts.
        #nominally 15 minnutes, make a bit longer to avoid attention.
        print "sleeping...."
        time.sleep(60*justincase)
        user = next(users)
        
        
    except StopIteration:
        break
    try:
        #print "Writing to JSON tweet number:"+str(count)
        text_value = user._json['text']
        language = user._json['lang']
        #print(text_value)
        print(language)
        
        if "RT" not in text_value:
            if language == "en":
                text[secondcount] = text_value
                secondcount = secondcount + 1
                print("current saved is:")
                print(secondcount)

    except UnicodeEncodeError:
        errorCount += 1
        print "UnicodeEncodeError,errorCount ="+str(errorCount)


print("Creating dataframe:")

d = {"text": text, "id": idvalues}
df = pd.DataFrame(data = d)

df.to_csv('upset.csv', header=True, index=False, encoding='utf-8')

print "completed"






#https://www.toptal.com/python/twitter-data-mining-using-python




#http://adilmoujahid.com/posts/2014/07/twitter-analytics/




#https://www.freelancer.com/community/articles/twitter-data-mining-a-guide-to-big-data-analytics-using-python




#https://www.quickstart.com/blog/post/how-to-use-twitter-for-data-mining/



