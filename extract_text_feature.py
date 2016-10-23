# imports
import os
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])


def makeVenueArray():

    f = open("./CS2108-Vine-Dataset/venue-name.txt")
    b = ['']*30

    for line in f:
        a = line.split("\t")
        b[int(a[0])-1] = a[1].strip("\n")

    return b


def makeFilenameArray(str):

    f = open("./CS2108-Vine-Dataset/vine-venue-"+str+".txt")
    g = open("./CS2108-Vine-Dataset/vine-desc-"+str+".txt")

    gdict = dict()
    
    descriptions = []
    venues = []

    for line in g:
        c = line.split("\t")
        gdict[c[0]] = c[1].strip("\n")


    count = 0

    for line in f:
        d = line.split("\t")
        venues.append(int(d[1].strip("\n"))-1)
        descriptions.append(gdict[d[0]])

    return venues, descriptions


def makeBagOfWords(arr):

    return vect.transform(arr)

venues = makeVenueArray()
tv, td = makeFilenameArray("training")
vv, vd = makeFilenameArray("validation")

text_clf = text_clf.fit(td, tv)


def classify(doc):

    d = [doc]

    prediction = text_clf.predict(d)

    #print venues[prediction[0]]

    return venues[prediction[0]]


def getClassificationReport():


    predicted = text_clf.predict(vd)

    print metrics.accuracy_score(vv, predicted)
    print(metrics.classification_report(vv, predicted, target_names=venues))
  
    
