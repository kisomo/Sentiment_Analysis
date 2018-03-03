import numpy as np

#http://textblob.readthedocs.io/en/dev/quickstart.html

from textblob import TextBlob

#wiki = TextBlob("Python is a high-level, general-purpose programming language.")

wiki = TextBlob("constant flux, as society view of what is social acceptable, and what is considered. To explore this concept, consider the following definition.")
print(wiki.tags)  # Parts of speach

print(wiki.noun_phrases)

print(wiki.sentiment)

print(wiki.sentiment.polarity)
print(wiki.sentiment.subjectivity)

print(wiki.words)

print(wiki.sentences)


for sentence in wiki.sentences:
    print(sentence.sentiment.polarity)
    print(sentence.sentiment.subjectivity)


#print(sentence.words)

print(wiki.words[2].singularize())

print(wiki.words[-1].pluralize())

#Words can be lemmatized by calling the lemmatize method.

from textblob import Word
w = Word("octopi")
print(w.lemmatize())

w = Word("went")
print(w.lemmatize("v")) # Pass in WordNet part of speech (verb)




