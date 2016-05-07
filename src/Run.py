#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
#from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
#from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    #    myStupidClassifier = StupidRecognizer(data.trainingSet,
    #                                          data.validationSet,
    #                                          data.testSet)
    # Uncomment this to make your Perceptron evaluated
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,  # 0.005
                                        epochs=100) # 30
    
    # Train the classifiers
    print("=========================")
    print("Training..")
    
    print("\nStart training the Perceptron..")
    myPerceptronClassifier.train()
    print("Done..")
    
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    perceptronPred = myPerceptronClassifier.evaluate()
    
    # Report the result
    print("=========================")
    print("Evaluation..")
    evaluator = Evaluator()
    
    print("\nResult of the Perceptron recognizer:")
    evaluator.printAccuracy(data.testSet, perceptronPred)
    evaluator.printConfusionMatrix(data.testSet, perceptronPred)
    evaluator.printClassificationResult(data.testSet, perceptronPred, None)

if __name__ == '__main__':
    main()
