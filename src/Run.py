#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bokeh.plotting import figure, output_file, show
import pandas
    
from data.mnist_seven import MNISTSeven
#from model.stupid_recognizer import StupidRecognizer
#from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator

def main():
    learningRate=0.05
    epochs=500
    
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    
    myLRClassifier = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=learningRate,
                                        epochs=epochs)
    
    # Train the classifiers
    print("=========================")
    print("Training..")
    
    
    print("\nLogistic Regression training..")
    result_training, result_validation = myLRClassifier.train()
    print("Done..")
    
    lrPred = myLRClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()
    
    
    print("\nResult of the Logistic Regression recognizer:")
    #evaluator.printComparison(data.testSet, lrPred)
    evaluator.printAccuracy(data.testSet, lrPred)
    
    visualize(result_training, result_validation, epochs, learningRate)


def visualize(result_training, result_validation, epochs, learningRate):
    title = "plot_%sepoches_lr%s.html"%(epochs,str(learningRate).replace(".",","))
    output_file(title)
    p = figure(width=800, height=800, title="Logistic neuron with %s epochs, learning rate = %s"%(epochs,learningRate))
    data_t = pandas.DataFrame.from_dict(result_training, orient="index")
    data_v = pandas.DataFrame.from_dict(result_validation, orient="index")
    
    p.line(list(data_t.index), data_t[0], color="red", legend="training")
    p.line(list(data_v.index), data_v[0], color="blue", legend="validation")
    show(p)

if __name__ == '__main__':
    main()
