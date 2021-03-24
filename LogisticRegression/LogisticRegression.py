import numpy as np

class LogisticRegression:
    def __init__(self, size, learnig_rate):
        self.weight = np.random.rand(size, 1)
        self.baias = 0
        self.learnig_rate = learnig_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        Y_prediction = np.zeros((1,len(x)))
        A = self.sigmoid(np.dot(self.weight.T, x) + self.baias)
        Y_prediction[A > 0.5] = 1.
        return Y_prediction

    def fit(self, X, y, ephocs=100):
        m = len(X)
        for i in range(ephocs):
            #fowardPass
            predict = np.dot(X, self.weight) + self.baias
            activation = self.sigmoid(predict)
            #cost = (-1/len(X))*np.sum(y*np.log(activation)+(1-y)*np.log(1-activation))
            #BackPass
            dw = (1/m)*np.dot(X.T, (activation-y))
            db = (1/m)*np.sum(activation-y)
            self.weight = self.weight - dw*self.learnig_rate
            self.baias = self.baias - db*self.learnig_rate
            