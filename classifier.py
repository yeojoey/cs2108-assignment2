from video import Video
import preprocess as pp

import numpy as np
import os
import cv2

counters = np.zeros(30)

def init(videoPath):
    video = pp.processVideo(videoPath)
    return video

def softmax(video,matrix,videos):
    fVector = video.featureVector

    #compare here


    maxIndex = 0
    for i in range(len(counters)):
        if counters[i] > counters[maxIndex]:
            maxIndex = i

    return maxIndex+1
    

def kNN(video,matrix,videos):
    fVector = video.featureVector

    #compare here


    maxIndex = 0
    for i in range(len(counters)):
        if counters[i] > counters[maxIndex]:
            maxIndex = i

    return maxIndex+1

    
def svm(video,matrix,videos):
    fVector = video.featureVector

    #compare here


    maxIndex = 0
    for i in range(len(counters)):
        if counters[i] > counters[maxIndex]:
            maxIndex = i

    return maxIndex+1
    
def linearRegression(video,matrix,videos):
    fVector = video.featureVector

    #compare here


    maxIndex = 0
    for i in range(len(counters)):
        if counters[i] > counters[maxIndex]:
            maxIndex = i

    return maxIndex+1
