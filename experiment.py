"""Cell division detection with Artificial Neural Networks"""
import pickle
import gzip

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from csxnet.datamodel import RData, CData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.brainforge.Utility.cost import Xent, MSE
from csxnet.thNets.thANN import ConvNet

ltpath = "/data/Prog/data/learning_tables/"
theano.config.exception_verbosity = "high"


class FFNetTheano:
    def __init__(self, data: CData, eta, lmbd):
        """Simple Feed Forward architecture implemented with Theano"""
        self.data = data

        inputs = data.data.shape[1]
        hiddens = 100

        l2term = 1 - ((eta * lmbd) / data.N)

        outputs = len(data.categories)

        X = T.matrix("X")  # Batch of inputs
        Y = T.vector("Y", dtype="int64")  # Batch of targets

        m = T.scalar(dtype="int64")  # Input batch size

        # Define the weights and biases of the layers
        W1 = theano.shared(np.random.randn(inputs, hiddens) / np.sqrt(inputs))
        W2 = theano.shared(np.random.randn(hiddens, outputs) / np.sqrt(hiddens))

        # Define the activations
        A1 = T.nnet.sigmoid(T.dot(X, W1))
        A2 = T.nnet.softmax(T.dot(A1, W2))

        # Cost function is cross-entropy
        xent = T.nnet.categorical_crossentropy(A2, Y).sum()
        prediction = T.argmax(A2, axis=1)

        # Define the weight update rules
        update_W1 = (l2term * W1) - ((eta / m) * theano.grad(xent, W1))
        update_W2 = (l2term * W2) - ((eta / m) * theano.grad(xent, W2))

        self._fit = theano.function((X, Y, m), updates=((W1, update_W1), (W2, update_W2)))
        self.predict = theano.function((X, Y), outputs=(xent, prediction))

    def train(self, epochs, batch_size):
        print("Start training")
        for epoch in range(1, epochs+1):
            for no, batch in enumerate(self.data.batchgen(batch_size)):
                m = batch[0].shape[0]
                questions, targets = batch[0], np.amax(batch[1], axis=1).astype("int64")
                self._fit(questions, targets, m)
            if epoch % 1 == 0:
                costt, acct = self.evaluate("testing")
                costl, accl = self.evaluate("learning")
                print("---- Epoch {}/{} Done! ----".format(epoch, epochs))
                print("Cost:", (costt + costl) / 2)
                print("AccT:", acct)
                print("AccL:", accl)

    def evaluate(self, on="testing"):
        m = self.data.n_testing
        deps = {"t": self.data.testing, "l": self.data.learning}[on[0]][:m]
        indeps = {"t": self.data.tindeps, "l": self.data.lindeps}[on[0]][:m]
        cost, answers = self.predict(deps, indeps)
        rate = np.mean(np.equal(answers, indeps))
        return cost, rate


class CNNetTheano(ConvNet):
    def __init__(self, data, eta, lmbd):
        nfilters = 1
        cfshape = (5, 5)
        pool = 2
        hidden_fc = 120
        ConvNet.__init__(self, data, eta, lmbd, nfilters, cfshape, pool, hidden_fc)


class FFNetThinkster(Network):
    def __init__(self, data, eta, lmbd):
        Network.__init__(self, data, eta, lmbd, Xent)
        self.add_fc(120)
        self.finalize_architecture()

    def train(self, epochs, batch_size):
        scores = [list(), list()]
        for epoch in range(1, epochs+1):
            self.learn(batch_size=batch_size)
            scores[0].append(self.evaluate())
            scores[1].append(self.evaluate("learning"))
            if epoch % 1 == 0:
                print("Epoch {}, Err: {}".format(epoch, self.error))
                print("Acc:", scores[0][-1], scores[1][-1])
        return scores


class CNNetThinkster(Network):
    def __init__(self, data, eta, lmbd):
        Network.__init__(self, data, eta, lmbd, Xent)
        self.add_conv((5, 5), n_filters=1)
        self.add_pool(2)
        # self.add_conv((3, 3), n_filters=3)
        # self.add_pool(2)
        # self.add_fc(100)
        self.add_fc(30)
        self.finalize_architecture()

    def train(self, epochs, batch_size):
        for epoch in range(epochs):
            self.learn(batch_size=batch_size)
            print("Epoch {}".format(epoch))
            print("Cost':", self.error)
            print("Acc:", self.evaluate(), self.evaluate("learning"))


if __name__ == '__main__':

    f = gzip.open(ltpath + "gorctrlt.pkl.gz", "rb")
    questions, targets = pickle.load(f)
    questions = questions.reshape(questions.shape[0], np.prod(questions.shape[1:]))
    f.close()

    np.divide(questions, 255, out=questions)
    np.equal(targets, True, targets)
    lt = questions.astype(float), targets.astype(int)

    myData = CData(lt, cross_val=0.2, header=None, pca=700)
    net = FFNetThinkster(myData, eta=0.1, lmbd=2.0)
    print("Initial test:", net.evaluate())
    score = net.train(100, 20)

    X = np.arange(len(score[0]))
    plt.plot(X, score[0], "b", label="T")
    plt.plot(X, score[1], "r", label="L")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
