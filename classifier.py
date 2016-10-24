__author__ = 'xiangwang1223@gmail.com'

import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import neighbors

# Trian your own classifier.
# Here is the simple implementation of SVM classification.
def myKNN(X_train,Y_train,X_test,Y_gnd):
    # 3. Generate the predicted label matrix Y_predicted for X_test via SVM or other classifiers.
    instance_num, class_num = Y_gnd.shape

    Y_predicted = np.asmatrix(np.zeros([instance_num, class_num]))

    # 4. Train the classifier.
    #model = svm.SVR(kernel='rbf', degree=3, gamma=0.01, shrinking=True, verbose=False, max_iter=-1)
    model = neighbors.KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, np.ravel(Y_train))
    Y_predicted = np.asmatrix(model.predict(X_test))
    print('SVM Train Done.')
    Y_gnd = np.ravel(Y_gnd)
    Y_predicted = map(int,np.ravel(Y_predicted))

    print metrics.accuracy_score(Y_gnd, Y_predicted)

    return Y_predicted


if __name__ == '__main__':
    mat_path = 'Sample.mat'
    output_path = 'Output.mat'

    mySVM(mat_path, output_path)
