from deeplearning.preprocessing.extract_audio import getAudioClip
from deeplearning.preprocessing.extract_frame import getKeyFrames
from deeplearning.featureextracting.acoustic.extract_acoustic import getAcousticFeatures
from video import Video

import numpy as np
import os
import cv2

videos = []
venues = {}
maxSize = 0

def processVideo(videoPath,fileName):
    global maxSize
    if os.path.isfile("./deeplearning/data/audio/"+fileName+".wav") == False :
        try:
            getAudioClip(videoPath,"./deeplearning/data/audio/"+fileName+".wav")
        except Exception:
            print Exception
        
    vidcap = cv2.VideoCapture(videoPath)
    keyframes = getKeyFrames(vidcap,"./deeplearning/data/frame/"+fileName+"-")
    vidcap.release()
    
    acousticFeature = getAcousticFeature("./deeplearning/data/audio/"+fileName+".wav")
    if acousticFeature.shape[0] > maxSize:
        maxSize = acousticFeature.shape[0]

    venue = getVenue(fileName)
    
    video = Video(fileName, acousticFeature, venue)

    
    return video
    
def preProcess(videoList,vidCount,venueFile):
    global maxSize
    global videos
    
    generateVenueDict(venueFile)

    maxSize = 0
    videos = []
    for file in videoList:
        print ("Processing:",file)
        fileName = file[file.rfind("/")+1:len(file)-4]
        video = processVideo(file,fileName)
        videos.append(video)
        
    print (maxSize)
    print (len(videos))
    x = np.zeros((vidCount,maxSize))
    y = np.zeros((vidCount,1))
    print (x.shape, y.shape)
    for row in range(len(videos)):
        vector = videos[row].featureVector
        y[row][0] = videos[row].venue
        for col in range(maxSize):
            x[row][col] = vector[col%len((vector)-1)]

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
    venues = {}
    inputs = [line.rstrip('\n') for line in open(venueFile)]
    for i in range(len(inputs)):
        tmp = inputs[i].split("\t")
        venues[tmp[0]] = tmp[1]
        
def getVenue(fileName):
    global venues
    venue = venues[fileName]
    print (fileName,venue)
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
