# imports
import process_groundtruths
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)

gt = process_groundtruths.gt
td = gt.training_desc
tv = gt.training_venue

def makeArray():

    arr = [''] * 31 # for number of venues

    
    for key, value in td.items():

        venue = tv[key]
        if arr[int(venue)] == "":
            arr[int(venue)] += value
        else:
            arr[int(venue)] = arr[int(venue)]+" "+value
        
    return arr


def makeBagOfWords(arr):

    return vectorizer.fit_transform(arr)


if __name__ == "__main__":

    a = makeArray()
    X = makeBagOfWords(a)
    
    # X is a bag of words matrix where the index = venue id
    # 0 is EMPTY
    #
    # print X.toarray()
    
