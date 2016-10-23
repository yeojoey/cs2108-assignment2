# imports
import os
import process_groundtruths
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)

gt = process_groundtruths.gt
td = gt.training_desc
tv = gt.training_venue

def makeVenueArray():

    arr = [' '] * 30

    for key, value in tv.items():
        arr[int(key)-1] = value

    return arr

def makeDocArray():

    arr = [''] * 30 # for number of venues

    
    for key, value in td.items():

        venue = tv[key]
        if arr[int(venue)-1] == "":
            arr[int(venue)-1] += value
        else:
            arr[int(venue)-1] = arr[int(venue)-1]+" "+value
        
    return arr


def makeBagOfWords(arr):

    return vectorizer.fit_transform(arr)


if __name__ == "__main__":

    
    v = makeVenueArray()
    a = makeDocArray()
    X = makeBagOfWords(a)
    
    # X is a bag of words matrix where the index = (venue id - 1)
    #
    # print X.toarray()
    
