#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven

from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                       one_hot=False)

    # Logistic Regression
    act = "softmax" # sigmoid softmax
    nlayer = 3  # 2, 3, 4 
    lr = 0.05 # 0.1, 0.05, 0.001
    decay = 0.99
    minibatch= None  # Not ready yet 
    # number of neurones in the hidden layers
    neuronHidden = 100
    myMLPClassifier = MultilayerPerceptron(data.training_set,
                                           data.validation_set,
                                           data.test_set,
                                           learning_rate=lr,
                                           epochs=100,
                                           output_activation=act,
                                           nlayer=nlayer,
                                           neuronHidden = neuronHidden,
                                           decay=decay,
                                           minibatch = minibatch)

    print("\n MLP Training..")
    myMLPClassifier.train()
    print("Done..")

    mlpPred = myMLPClassifier.evaluate()

    # Report the result #
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the Multi-layer Perceptron recognizer (on test set):")
    evaluator.printAccuracy(data.test_set, mlpPred)

    # Draw
    plot = PerformancePlot("MLP-%s-%slayers-%slr-%s#hidden-%sdecay"%(act,nlayer, lr,neuronHidden,decay))
    plot.draw_performance_epoch(myMLPClassifier.performances,
                                myMLPClassifier.epochs)

if __name__ == '__main__':
    main()
