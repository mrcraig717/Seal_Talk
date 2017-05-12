from sklearn.linear_model import LogisticRegression
import numpy as np
################################################
#Class made for color classification.
################################################

class seallogit(LogisticRegression):

    def __init__(self):
        super(seallogit, self).__init__()

    def predict(self, X):
        ##########OverRided function from Logistic Regression
        ##########Takes a BGR Vector and changes to our feature vector
        predictorShape = np.shape(X)
        X[X < 1.0] = 1.0
        X = np.hstack((X, (X[:, 0] / X[:, 1]).reshape((predictorShape[0], 1)),
                          (X[:, 0] / X[:, 2]).reshape((predictorShape[0], 1)),
                          (X[:, 1] / X[:, 2]).reshape(predictorShape[0], 1)))
        P = []
        return super(seallogit, self).predict(X)

    def fit(self, X, y, sample_weight=None):
        ##########Overrided Function from LogisticRegression just changes 
        ########## a BGR vector to the proper feature vector
        predictorShape = np.shape(X)
        X[X < 1.0] = 1.0
        X = np.hstack((X, (X[:, 0] / X[:, 1]).reshape((predictorShape[0], 1)),
                          (X[:, 0] / X[:, 2]).reshape((predictorShape[0], 1)),
                          (X[:, 1] / X[:, 2]).reshape(predictorShape[0], 1)))
        super(seallogit, self).fit(X, y)

    def predictP(self, X):
        #####DITTO above returns Probability of class one not used at the moment
        predictorShape = np.shape(X)
        X[X < 1.0] = 1.0
        X = np.hstack((X, (X[:, 0] / X[:, 1]).reshape((predictorShape[0], 1)),
                          (X[:, 0] / X[:, 2]).reshape((predictorShape[0], 1)),
                          (X[:, 1] / X[:, 2]).reshape(predictorShape[0], 1)))

        return super(seallogit, self).predict_proba(X)[:,0]



