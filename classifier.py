from video import Video
import preprocess as pp

import numpy as np
import os
import cv2
import scipy.io as sio
from sklearn import svm
from sklearn.metrics import classification_report

counters = np.zeros(30)
venues = {}

def init(videoPath):
    video = pp.processVideo(videoPath)
    return video

def generateVenueDict():
    global venues
    venues = {}
    inputs = [line.rstrip('\n') for line in open("./CS2108-Vine-Dataset/venue-name.txt")]
    for i in range(len(inputs)):
        tmp = inputs[i].split("\t")
        venues[tmp[0]] = tmp[1]
    

def svm(X_train,Y_train,X_test,Y_gnd):
    global venues
    generateVenueDict()

    instance_num, class_num = Y_gnd.shape

    Y_predicted = np.asmatrix(np.zeros([instance_num, class_num]))

    # 4. Train the classifier.
    model = svm.SVR(kernel='rbf', degree=3, gamma=0.1, shrinking=True, verbose=False, max_iter=-1)
    model.fit(X_train, Y_train)
    Y_predicted = np.asmatrix(model.predict(X_test))


def softmax(video,matrix,videos):
    global venues
    generateVenueDict()
    fVector = video.featureVector

    #compare here


    maxIndex = 0
    for i in range(len(counters)):
        if counters[i] > counters[maxIndex]:
            maxIndex = i
    
    return venues[str(maxIndex+1)]
    

def kNN(video,matrix,videos):
    global venues
    generateVenueDict()
    fVector = video.featureVector

    #compare here


    maxIndex = 0
    for i in range(len(counters)):
        if counters[i] > counters[maxIndex]:
            maxIndex = i

    return venues[str(maxIndex+1)]

    
def linearRegression(video,matrix,videos):
    global venues
    generateVenueDict()
    fVector = video.featureVector

    #compare here


    maxIndex = 0
    for i in range(len(counters)):
        if counters[i] > counters[maxIndex]:
            maxIndex = i

    return venues[str(maxIndex+1)]
