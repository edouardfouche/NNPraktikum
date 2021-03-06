#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven

from model.denoising_ae import DenoisingAutoEncoder
from model.mlp import MultilayerPerceptron 

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot
from report.weight_visualization_plot import WeightVisualizationPlot

def main():
    # data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
    #                  one_hot=True, target_digit='7')

    # NOTE:
    # Comment out the MNISTSeven instantiation above and
    # uncomment the following to work with full MNIST task
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                      one_hot=False)

    # NOTE:
    # Other 1-digit classifiers do not make sense now for comparison purpose
    # So you should comment them out, let alone the MLP training and evaluation

    # Train the classifiers #
    print("=========================")
    print("Training the autoencoder..")

    myDAE = DenoisingAutoEncoder(data.training_set,
                                 learning_rate=0.05,
                                 epochs=30, decay=0.99)

    print("\nAutoencoder  has been training..")
    myDAE.train()
    print("Done..")

    # Multi-layer Perceptron
    # NOTES:
    # Now take the trained weights (layer) from the Autoencoder
    # Feed it to be a hidden layer of the MLP, continue training (fine-tuning)
    # And do the classification

    # reload data, simpler since it was modified by the autoencoder (adding the biais)
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                      one_hot=False)

    # Correct the code here
    myMLPClassifier = MultilayerPerceptron(data.training_set,
                                           data.validation_set,
                                            data.test_set,
                                            learning_rate=0.05,
                                            decay=0.99,
                                           epochs=30,
                                           input_weights=myDAE.layers[0].weights)

    print("\nMulti-layer Perceptron has been training..")
    myMLPClassifier.train()
    print("Done..")
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()

    # Report the result #
    print("=========================")
    evaluator = Evaluator()

    # print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    # evaluator.printAccuracy(data.test_set, stupidPred)

    # print("\nResult of the Perceptron recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.test_set, perceptronPred)

    # print("\nResult of the Logistic Regression recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.test_set, lrPred)

    print("\nResult of the DAE + MLP recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.test_set, mlpPred)

    # Draw
    plot = PerformancePlot("DAE + MLP on MNIST task")
    plot.draw_performance_epoch(myMLPClassifier.performances,
                                myMLPClassifier.epochs)
                                
    plotweights = WeightVisualizationPlot(myDAE.layers[0].weights)
    plotweights.show()

if __name__ == '__main__':
    main()
