
import matplotlib.pyplot as plt
import numpy as np
import time

class WeightVisualizationPlot(object):
    '''
    Here to implement the visualization of the weights
    '''

    def __init__(self, weights):
        self.weights = weights
        pass
    
    def show(self):
        weights = np.delete(self.weights, 0, 0) # delete biais
    
        fig, axes = plt.subplots(10, 10, figsize=(10,10))
        for i,ax in zip(range(weights.shape[1]),axes.ravel()):
            x = weights[:,i]
            x = x.reshape((28,28))
            ax.matshow(x, cmap="Greys")
            ax.axis('off') # avoid showing axis, not interesting
            
        date = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
        plt.savefig("AE_weights_visualization"+date+".png")
        plt.show()