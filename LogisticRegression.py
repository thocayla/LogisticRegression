import numpy as np

class LogisticRegression(object):
    def __init__(self, num_iter=100, fit_intercept=False):
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return (np.concatenate((intercept, X), axis=1))
    
    # Compute logistic function
    def sigmoid(self, a):
        return (1 / (1 + np.exp(-a)))
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.intercept(X)
            
        # weights initialization
        self.coef_ = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            a = np.dot(X, self.coef_)
            g = self.sigmoid(a)
            # minimize the loss function by decreasing the weights
            gradient = np.dot(X.T, (g - y)) / y.shape[0]
            self.coef_ -= gradient
    
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.intercept(X)
        return (self.sigmoid(np.dot(X, self.coef_)))

    def predict(self, X, threshold):
        return (self.predict_prob(X) >= threshold)