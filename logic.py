import preprocess as pp
import classifier
import numpy as np
import os
import scipy.io as sio

venues = {}

def preprocess():
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
    
def predict(paths,X_train,Y_train):
    global venues
    generateVenueDict()
    videoList = []
    videoList.append(paths)
    X_test, Y_gnd = pp.preProcess(videoList,1,"./CS2108-Vine-Dataset/vine-venue-validation.txt")
    Y_predicted = classifier.svm(X_train,Y_train,X_test,Y_gnd)

    print (Y_predicted.shape)
    
def generateVenueDict():
    global venues
    venues = {}
    inputs = [line.rstrip('\n') for line in open("./CS2108-Vine-Dataset/venue-name.txt")]
    for i in range(len(inputs)):
        tmp = inputs[i].split("\t")
        venues[tmp[0]] = tmp[1]

X_train, Y_train = preprocess()

for i in range(Y_train.shape[0]):
    print (Y_train[i][0],venues[Y_train[i][0]])
