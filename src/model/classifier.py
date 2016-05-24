# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from util.time import timed


class Classifier:
    """
    Abstract class of a classifier
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    @timed
    def train(self, trainingSet, validationSet):
        # train procedures of the classifier
        pass

    @abstractmethod
    @timed
    def classify(self, testInstance):
        # classify an instance given the model of the classifier
        pass

    @abstractmethod
    @timed
    def evaluate(self, test):
        # evaluate a whole test set given the model of the classifier
        pass
