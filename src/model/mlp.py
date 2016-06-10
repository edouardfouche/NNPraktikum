
import numpy as np
import random

# from util.activation_functions import Activation
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from sklearn.metrics import accuracy_score
#from loss_functions import CrossEntropyError


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='sigmoid',
                 cost='crossentropy', learning_rate=0.01, epochs=50, neuronHidden=50, nlayer=2, decay=0.99,
                 minibatch= None):

        """
        A digit-7 recognizer based on logistic regression algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost = cost

        self.training_set = train
        self.validation_set = valid
        self.test_set = test
        self.decay = decay
        self.minibatch = minibatch

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights

        self.layers = []
        self.layers.append(LogisticLayer(train.input.shape[1], neuronHidden, None, activation='sigmoid',
                                   is_classifier_layer=False, is_input_layer=True))
        if nlayer > 2:
            for n in range(0,nlayer-2):  
                self.layers.append(LogisticLayer(neuronHidden, neuronHidden, None, activation='sigmoid',
                                   is_classifier_layer=False, is_input_layer=False))
                                   
        self.layers.append(LogisticLayer(neuronHidden, 10, None, output_activation, 
                                         is_classifier_layer=True, is_input_layer=False))
        # 100 100 great ! 93
        self.outp = None
        
        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)
        
        print("Created network:")
        print("- %s layer(s)"%nlayer)
        print("- %s neurons in hidden layer(s)"%neuronHidden)
        print("- %s as output activation"%output_activation)
        print("- %s as learning rate, with %s decay"%(learning_rate,decay))
        

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self.get_layer(0)

    def _get_output_layer(self):
        return self.get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        self.outp = inp
        for i,layer in enumerate(self.layers):
            self.outp = layer.forward(self.outp)
            
    def _encode_target(self, target):
        if target == 0:
            val = [1,0,0,0,0,0,0,0,0,0]
        elif target == 1:
            val = [0,1,0,0,0,0,0,0,0,0]
        elif target == 2:
            val = [0,0,1,0,0,0,0,0,0,0]
        elif target == 3:
            val = [0,0,0,1,0,0,0,0,0,0]
        elif target == 4:
            val = [0,0,0,0,1,0,0,0,0,0]
        elif target == 5:
            val = [0,0,0,0,0,1,0,0,0,0]
        elif target == 6:
            val = [0,0,0,0,0,0,1,0,0,0]
        elif target == 7:
            val = [0,0,0,0,0,0,0,1,0,0]
        elif target == 8:
            val = [0,0,0,0,0,0,0,0,1,0]
        elif target == 9:
            val = [0,0,0,0,0,0,0,0,0,1]
        return val
        
    def _decode_target(self, outp):
        m = max(outp)
        if outp[0] == m:
            val = 0
        elif outp[1] == m:
            val = 1
        elif outp[2] == m:
            val = 2
        elif outp[3] == m:
            val = 3
        elif outp[4] == m:
            val = 4 
        elif outp[5] == m:
            val = 5 
        elif outp[6] == m:
            val = 6 
        elif outp[7] == m:
            val = 7 
        elif outp[8] == m:
            val = 8 
        elif outp[9] == m:
            val = 9 
        
        return val

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        target = self._encode_target(target)
        
        if self.minibatch:
            for layer in self.layers:
                layer.deltas = layer.deltas / self.minibatch
        
        lastlayer = self.layers[-1]

        lastlayer.computeDerivative(target)

        for layer,nextlayer in zip(self.layers[::-1][1:], self.layers[::-1][:-1]):         
            layer.computeDerivative(nextlayer.deltas,nextlayer.weights)

    def _update_weights(self, learning_rate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(learning_rate)
            
    def _init_deltas(self):
        for layer in self.layers:
            layer.deltas = np.zeros((1, layer.n_out))

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        train = []
        for img, label in zip(self.training_set.input, self.training_set.label):
            train.append((img, label))
            
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch(train)
            
            if verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("Learning rate: %s"%self.learning_rate)
                print("-----------------------------")
            
            self.learning_rate = self.learning_rate * self.decay
                

    def _train_one_epoch(self, train):
        """
        Train one epoch, seeing all input instances
        """
        # array_split
        #import pdb ; pdb.set_trace()
            
        # Shuffle the training set
        np.random.shuffle(train)
        
        for img, label in train:
            # Do a forward pass to calculate the output and the error
            #sself._init_deltas()
            self._feed_forward(img)
            
            self._compute_error(label)

            self._update_weights(self.learning_rate)
        pass

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        self._feed_forward(test_instance)
        outp = self._decode_target(self.outp)
        return outp

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
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
