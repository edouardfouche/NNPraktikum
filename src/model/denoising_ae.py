# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from model.logistic_layer import LogisticLayer
from model.auto_encoder import AutoEncoder
from util.loss_functions import MeanSquaredError

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class DenoisingAutoEncoder(AutoEncoder):
    """
    A denoising autoencoder.
    """

    def __init__(self, train, neuronHidden=100, learning_rate=0.1, epochs=30, decay=0.99):
        """
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

        self.training_set = train
        self.layers = []
        self.decay = decay 
        
        self.layers.append(LogisticLayer(train.input.shape[1],
                                 neuronHidden, None,
                                 activation="sigmoid",
                                 is_classifier_layer=False))

        # Output layer
        self.layers.append(LogisticLayer(neuronHidden,
                                         train.input.shape[1]+1, None,
                                         activation="sigmoid",
                                         is_classifier_layer=False))
        
        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
                                            
        self.error_function = MeanSquaredError()

    def _get_layer(self, layer_index):
        return self.layers[layer_index]
        
    def _get_output_layer(self):
        return self._get_layer(-1)
        
    def train(self, verbose=True):
        """
        Train the denoising autoencoder
        """
        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            epoch_error = self._train_one_epoch()
            
            if verbose:
                print("Mean error over epoch: %s"%(epoch_error))
                print("-----------------------------")
                print("Learning rate: %s"%self.learning_rate)
                print("-----------------------------")
            
            self.learning_rate = self.learning_rate * self.decay
        pass

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        epoch_error = []
        for img in self.training_set.input:
            # Normalize the image 
            
            #import pdb ; pdb.set_trace()
            #norm_img = (img - img.argmin())/(img.argmax() - img.argmin())
            
            self._feed_forward(img)
            self._compute_deltas(img)
            self._update_weights()
            
            epoch_error.append(self._compute_error(img))
        
        return np.mean(epoch_error)
    
    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer
        """

        # Feed forward layer by layer
        # The output of previous layer is the input of the next layer
        last_layer_output = inp

        for layer in self.layers:
            last_layer_output = layer.forward(last_layer_output)
            # Do not forget to add bias for every layer
            last_layer_output = np.insert(last_layer_output, 0, 1, axis=0)
    
    def _compute_deltas(self, img):
        # Get output layer
        output_layer = self._get_output_layer()

        # Calculate the deltas of the output layer
        output_layer.deltas = (img - output_layer.outp)*output_layer.outp*(1 - output_layer.outp)

        # Calculate deltas (error terms) backward except the output layer
        for i in reversed(range(0, len(self.layers) - 1)):
            current_layer = self._get_layer(i)
            next_layer = self._get_layer(i+1)
            next_weights = np.delete(next_layer.weights, 0, axis=0)
            next_derivatives = next_layer.deltas

            current_layer.computeDerivative(next_derivatives, next_weights.T)
            
    def _compute_error(self, img):
        return self.error_function.calculate_error(img, self._get_output_layer().outp)
            
    def _update_weights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        # Update the weights layer by layers
        for layer in self.layers:
            layer.updateWeights(self.learning_rate)

    def _get_weights(self):
        """
        Get the weights (after training)
        """
        return [self.layers[0].weights, self.layers[1].weights]

        pass
