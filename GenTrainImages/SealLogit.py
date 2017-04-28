from sklearn.linear_model import LogisticRegression
import numpy as np


class seallogit(LogisticRegression):

    def __init__(self):
        super(SealLogit, self).__init__()

    def predict(self, X):
        predictorShape = np.shape(X)
        X[X < 1.0] = 1.0
        X = np.hstack((X, (X[:, 0] / X[:, 1]).reshape((predictorShape[0], 1)),
                          (X[:, 0] / X[:, 2]).reshape((predictorShape[0], 1)),
                          (X[:, 1] / X[:, 2]).reshape(predictorShape[0], 1)))
        P = []
        return super(SealLogit, self).predict(X)

    def fit(self, X, y, sample_weight=None):
        predictorShape = np.shape(X)
        X[X < 1.0] = 1.0
        X = np.hstack((X, (X[:, 0] / X[:, 1]).reshape((predictorShape[0], 1)),
                          (X[:, 0] / X[:, 2]).reshape((predictorShape[0], 1)),
                          (X[:, 1] / X[:, 2]).reshape(predictorShape[0], 1)))
        super(SealLogit, self).fit(X, y)

    def predictP(self, X):

        predictorShape = np.shape(X)
        X[X < 1.0] = 1.0
        X = np.hstack((X, (X[:, 0] / X[:, 1]).reshape((predictorShape[0], 1)),
                          (X[:, 0] / X[:, 2]).reshape((predictorShape[0], 1)),
                          (X[:, 1] / X[:, 2]).reshape(predictorShape[0], 1)))

        if super(seallogit, self).predict_proba(X)[0] > .4:
            return 1.0
        else:
            return 0,0
        #return super(SealLogit, self).predict_proba(X)[0]

