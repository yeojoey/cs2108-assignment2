import preprocess as pp
import classifier
import numpy as np
import os

X_train = []
Y_train = []
venues = {}

def preprocess():
    global X_train
    global Y_train

    generateVenueDict()
    X_train , Y_train = pp.preProcess("./CS2108-Vine-Dataset/vine/training",3000,"./CS2108-Vine-Dataset/vine-venue-training.txt")
    print (X_train,shape),(Y_train.shape)
    
def predict(paths,):
 
    Y_predicted = classifier.svm(X_train,Y_train,X_test,Y_gnd)
    
def generateVenueDict():
    global venues
    venues = {}
    inputs = [line.rstrip('\n') for line in open("./CS2108-Vine-Dataset/venue-name.txt")]
    for i in range(len(inputs)):
        tmp = inputs[i].split("\t")
        venues[tmp[0]] = tmp[1]

preprocess()
