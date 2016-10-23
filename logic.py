import preprocess as pp
from classifier import mySVM
import numpy as np
import os
import scipy.io as sio

venues = {}

def preprocess():
    generateVenueDict()
    if os.path.isfile("./processed.mat") == False :
        videoList = os.listdir("./CS2108-Vine-Dataset/vine/training")
        for i in range(len(videoList)):
            videoList[i] = "./CS2108-Vine-Dataset/vine/training/"+videoList[i]
        X_train , Y_train = pp.preProcess(videoList,3000,"./CS2108-Vine-Dataset/vine-venue-training.txt")
        sio.savemat("./processed.mat", {'X_train': X_train, 'Y_train': Y_train})
    else:
        mat_contents = sio.loadmat("./processed.mat")
        X_train = np.asmatrix(mat_contents['X_train'])
        Y_train = np.asmatrix(mat_contents['Y_train'])
        
    print (X_train.shape),(Y_train.shape)
    return X_train , Y_train
    
def predict(X_train,Y_train,X_test,Y_gnd):
    global venues
    generateVenueDict()    
    X_test_temp = np.zeros((1,X_train.shape[1]))
    for i in range(X_train.shape[1]):
        X_test_temp[0][i] = X_test[0][i%X_test.shape[1]]
    X_test = X_test_temp
    print (X_test.shape)
    
    Y_predicted = mySVM(X_train,Y_train,X_test,Y_gnd)

    print (venues[str(Y_predicted[0])])
    return venues[str(Y_predicted[0])]

def processFile(paths):
    videoList = []
    videoList.append(paths)
    X_test, Y_gnd = pp.preProcess(videoList,1,"./CS2108-Vine-Dataset/vine-venue-validation.txt")
    return X_test, Y_gnd

def generateVenueDict():
    global venues
    venues = {}
    inputs = [line.rstrip('\n') for line in open("./CS2108-Vine-Dataset/venue-name.txt")]
    for i in range(len(inputs)):
        tmp = inputs[i].split("\t")
        venues[tmp[0]] = tmp[1]

#X_train,Y_train = preprocess()
#X_test,Y_gnd = processFile("./CS2108-Vine-Dataset/vine/validation/1000861821491707904.mp4")
#predict(X_train,Y_train,X_test,Y_gnd)
