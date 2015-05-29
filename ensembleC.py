from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class EnsembleClassifier:
    def __init__(self):
        self.classifiers = []
        # Pair [classifier,weight]
        self.classifiers.append([RandomForestClassifier(), 0.5])
        self.classifiers.append([LogisticRegression(C=0.1, penalty='l1'), 0.5])

    def fit(self, X, y):
        for c in self.classifiers:
            classifier = c[0]
            classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        preds = [[0,0] for x in X]
        totW = 0;
        for c in self.classifiers:
            totW = totW + c[1]
        
        for c in self.classifiers:
            classifier = c[0]
            w = c[1]
            predTemp = classifier.predict_proba(X)
            for i in range(len(X)):
                for j in range(len(preds[i])):
                    preds[i][j] = preds[i][j]+ w*predTemp[i][j]/totW  
        return preds;
