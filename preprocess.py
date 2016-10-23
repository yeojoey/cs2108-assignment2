from deeplearning.preprocessing.extract_audio import getAudioClip
from deeplearning.preprocessing.extract_frame import getKeyFrames
from deeplearning.featureextracting.acoustic.extract_acoustic import getAcousticFeatures
from video import Video

import numpy as np
import os
import cv2

videos = []
venues = {}
minSize = 0

def processVideo(videoPath,fileName):
    global minSize
    getAudioClip(videoPath,"./deeplearning/data/audio"+fileName+".wav")
    #vidcap = cv2.VideoCapture(videoPath)
    #keyframes = getKeyFrames(vidcap,"./deeplearning/data/frame"+fileName+"-")
    #vidcap.release()
    
    acousticFeature = getAcousticFeature("./deeplearning/data/audio"+fileName+".wav")
    if acousticFeature.shape[0] < minSize:
        minSize = acousticFeature.shape[0]

    venue = getVenue(fileName)
    
    video = Video(fileName, acousticFeature, venue)

    return video
    
def preProcess(videoPath,vidCount,venueFile):
    global minSize
    global videos
    generateVenueDict(venueFile)

    minSize = 0
    videos = []
    for file in os.listdir(videoPath):
        fileName = file[:len(file)-4]
        video = processVideo(videoPath+file,fileName)
        videos.append(video)

    x = np.zeros((vidCount,minSize))
    y = np.zeros((vidCount,1))
    for row in range(len(videos)):
        vector = videos[row].featureVector
        y[row][1] = videos[row].venue
        for col in range(len(vector)):
            x[row][col] = vector[col]

    return x, y

def getAcousticFeature(audioPath):
    feature_mfcc, feature_spect, feature_zerocrossing, feature_energy = getAcousticFeatures(audioPath)
    feature_mfcc = mat2vec(feature_mfcc)
    feature_spect = mat2vec(feature_spect)
    feature_zerocrossing = mat2vec(feature_zerocrossing)
    feature_energy = mat2vec(feature_energy)

    finalVec = np.append(feature_mfcc,feature_spect)
    finalVec = np.append(finalVec,feature_zerocrossing)
    finalVec = np.append(finalVec,feature_energy)

    return finalVec

def generateVenueDict(venueFile):
    global venues
    inputs = [line.rstrip('\n') for line in open(venueFile)]
    for i in range(len(inputs)):
        tmp = inputs[i].split("\t")
        venues[tmp[0]] = tmp[1]
        
def getVenue(fileName):
    global venues
    venue = venues[fileName]
    return venue

            
def mat2vec(mat):
    vec = np.zeros(mat.shape[1])
    for c in range(mat.shape[1]):
        total = 0
        for r in range(mat.shape[0]):
            total += mat[r,c]
        
        avg = total/mat.shape[1]
        vec[c] = avg
        
    return vec
