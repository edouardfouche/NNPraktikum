# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from sklearn.metrics import accuracy_score

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.net = LogisticLayer(784,1,is_classifier_layer=True)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        # Here you have to implement training method "epochs" times
        # Please using LogisticLayer class
        result_training = dict()
        result_validation = dict()
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs), end="\r")

            self._train_one_epoch()

            accuracy_training = accuracy_score(self.trainingSet.label,
                                     self.evaluate(self.trainingSet))
            accuracy_validation = accuracy_score(self.validationSet.label,
                                     self.evaluate(self.validationSet))
            result_training[epoch+1] = accuracy_training
            result_validation[epoch+1] = accuracy_validation
            
        return (result_training, result_validation)
    
    
    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.trainingSet.input, self.trainingSet.label):
            self.net.forward(np.append(img,1))
            self.net.computeDerivative(labels=label)
            self.net.updateWeights(self.learningRate)
        # if we want to do batch learning, accumulate the error
        # and update the weight outside the loop

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """

        # Here you have to implement classification method given an instance
        return self._fire(np.append(testInstance,1))


    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))
        
    def _fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # I already implemented it for you to see how you can work with numpy
        return self.net._fire(input)
