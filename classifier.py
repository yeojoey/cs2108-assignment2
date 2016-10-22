from video import Video
import preprocess as pp

import numpy as np
import os
import cv2

counters = np.zeros(30)
venues = {}

def init(videoPath):
    video = pp.processVideo(videoPath)
    return video

def generateVenueDict():
    global venues
    inputs = [line.rstrip('\n') for line in open("./CS2108-Vine-Dataset/venue-name.txt")]
    for i in range(len(inputs)):
        tmp = inputs[i].split("\t")
        venues[tmp[0]] = tmp[1]
    
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

    
def svm(video,matrix,videos):
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
