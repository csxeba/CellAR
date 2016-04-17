import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

import numpy as np


class CNN:
    def __init__(self, data, eta: float, lmbd: float, nfilters: int, conv: int, pool: int, hiddens: tuple):
        self.data = data
        self.eta = eta

        assert self.data.data.shape[2] == self.data.data.shape[3], "Non-Sqare images not <yet> supported"

        channels, inshape = self.data.data.shape[1], (self.data.data.shape[2], self.data.shape[2])
        coutshape = (inshape[0] - conv + 1), (inshape[1] - conv + 1), nfilters
        poutshape = (coutshape[0] // pool), (coutshape[1] // pool), nfilters

        fc_fanin = np.prod(poutshape)


