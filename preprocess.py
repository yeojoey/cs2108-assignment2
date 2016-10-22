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
    getAudioClip(videoPath,"./deeplearning/data/audio"+fileName+".wav")
    #vidcap = cv2.VideoCapture(videoPath)
    #keyframes = getKeyFrames(vidcap,"./deeplearning/data/frame"+fileName+"-")
    #vidcap.release()
    
    acousticFeature = getAcousticFeature("./deeplearning/data/audio"+fileName+".wav")
    if acousticFeature.shape[0] > maxSize:
        maxSize = acousticFeature.shape[0]

    venue = getVenue(fileName)
    
    video = Video(fileName, acousticFeature, venue)

    return video
    
def preProcess():
    global maxSize
    
    for file in os.listdir("./CS2108-Vine-Dataset/vine/training"):
        fileName = file[:len(file)-4]
        video = processVideo("./CS2108-Vine-Dataset/vine/training/"+file,fileName)
        videos.append(video)

    matrix = np.zeros((3000,maxSize))
    for row in range(len(videos)):
        vector = videos[row].featureVector
        for col in range(len(vector)):
            matrix[row][col] = vector[col]

    return matrix, videos

def getAcousticFeature(audioPath):
    feature_mfcc, feature_spect, feature_zerocrossing, feature_energy = getAcousticFeatures(audioPath)
    feature_mfcc = mat2vec(feature_mfcc)
    feature_spect = mat2vec(feature_spect)
    feature_zerocrossing = mat2vec(feature_zerocrossing)
    feature_energy = mat2vec(feature_energy)
    finalVec = np.concatenate(feature_mfcc,feature_spect,feature_zerocrossing,feature_energy)

    return finalVec

def generateVenueDict():
    inputs = [line.rstrip('\n') for line in open("./CS2108-Vine-Dataset/vine-venue-training.txt")]
    for i in range(len(inputs)):
        tmp = inputs[i].split("\t")
        venues[tmp[0]] = tmp[1]
        
def getVenue(fileName):
    generateVenueDict()
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
