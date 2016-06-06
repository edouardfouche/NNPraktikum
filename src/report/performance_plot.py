#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time


class PerformancePlot(object):
    '''
    Class to plot the performances
    Very simple and badly-styled
    '''

    def __init__(self, name):
        '''
        Constructor
        '''
        self.name = name

    def draw_performance_epoch(self, performances, epochs):
        plt.plot(range(epochs), performances, 'k',
                 range(epochs), performances, 'ro')
        plt.title("Performance of " + self.name + " over the epochs")
        plt.ylim(ymax=1)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        date = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
        plt.savefig(self.name+date+".png")
        plt.show()
        
        #return plt
