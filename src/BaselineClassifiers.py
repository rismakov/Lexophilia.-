from random import random
import numpy as np


class BaselineClassifier(object):

    def __init__(self, percentage=None, length=None):
        self._percentage = percentage
        self._len = length

    def fit(self, X, y):
        self._percentage = len([y==1])/len(y)

    def predict(self, X):
        predictions = []
        for i in xrange(self._len):
            rand = random()

            if rand <= self._percentage:
                predictions.append(1)
            if rand > self._percentage:
                predictions.append(0)

        return np.array(predictions)


class MajorityClassifier(object):

    def __init__(self, majority=None, length=None):
        self._majority = majority
        self._len = length

    def fit(self, X, y):
        if len(y==1) > len(y==0):
            self._majority = 1
        else:
            self._majority = 0

    def predict(self, X):
        return np.ones(self._len * self._majority)


class SmartBaseline(object):

    def __init__(self, male_percentage_by_year):
        self._prob = male_percentage_by_year

    def predict(self, X):
        predictions = []
        for i, yr in enumerate(X['year']):
            probability_male = self._prob[yr]
            rand = random()

            if rand <= probability_male:
                predictions.append(1)
            if rand > probability_male:
                predictions.append(0)

        return np.array(predictions)
