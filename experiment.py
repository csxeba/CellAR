"""Cell division detection with Artificial Neural Networks"""
import pickle
import gzip
import sys

import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from csxnet.datamodel import RData, CData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.brainforge.Utility.cost import Xent, MSE
from csxnet.brainforge.Utility.activations import *
from csxnet.thNets.thANN import ConvNetExplicit, ConvNetDynamic


ltpath = "/data/Prog/data/learning_tables/" if sys.platform != "win32" else "D:/Data/learning_tables/"
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
        Y = T.matrix("Y", dtype="int64")  # Batch of targets !! in vector form !!

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
        self.predict = theano.function(X, outputs=(prediction,))

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
        preds = self.predict(deps)
        rate = np.mean(np.equal(preds, indeps))
        return rate


class CNNexplicit(ConvNetExplicit):
    def __init__(self, data, eta, lmbd, cost):
        nfilters = 2
        cfshape = (3, 3)
        pool = 2
        hidden1 = 60
        hidden2 = 30
        cost = "MSE" if cost is MSE else "Xent"
        ConvNetExplicit.__init__(self, data, eta, lmbd,
                                 nfilters, cfshape, pool,
                                 hidden1, hidden2,
                                 cost)

    def train(self, epochs, batch_size):
        scores = [list(), list()]
        for epoch in range(1, epochs+1):
            self.learn(batch_size)
            if epoch % 1 == 0:
                tcost, tscore = self.evaluate("testing")
                lcost, lscore = self.evaluate("learning")
                scores[0].append(tscore)
                scores[1].append(lscore)
                print("Epoch {}/{} done! Cost: {}".format(epoch, epochs, lcost))
                print("T: {}\tL: {}".format(scores[0][-1], scores[1][-1]))

        return scores


class CNNdynamic(ConvNetDynamic):
    def __init__(self, data, eta, lmbd, cost):
        cost = "MSE" if cost is MSE else "Xent"
        ConvNetDynamic.__init__(self, data, eta, lmbd, cost)
        self.add_convpool(conv=3, filters=2, pool=2)
        self.add_fc(neurons=60)
        self.add_fc(neurons=30)
        self.finalize()

    def train(self, epochs, batch_size):
        scores = [list(), list()]
        for epoch in range(1, epochs + 1):
            self.learn(batch_size)
            if epoch % 1 == 0:
                tcost, tscore = self.evaluate("testing")
                lcost, lscore = self.evaluate("learning")
                scores[0].append(tscore)
                scores[1].append(lscore)
                print("Epoch {}/{} done! Cost: {}".format(epoch, epochs, lcost))
                print("T: {}\tL: {}".format(scores[0][-1], scores[1][-1]))

        return scores


class FFNetThinkster(Network):
    def __init__(self, data, eta, lmbd):
        Network.__init__(self, data, eta, lmbd, cost)
        for h in hiddens:
            if not drop and h:
                self.add_fc(h, activation=act_fn_H)
            if drop and h:
                self.add_drop(h, drop, activation=act_fn_H)
        self.finalize_architecture()

    def train(self, epochs, batch_size):
        scores = [list(), list()]
        for epoch in range(1, epochs+1):

            self.learn(batch_size=batch_size)

            if epoch % 2 == 0:
                scores[0].append(self.evaluate())
                scores[1].append(self.evaluate("learning"))
                print("Epoch {}, Err: {}".format(epoch, self.error))
                print("Acc:", scores[0][-1], scores[1][-1])

        return scores


def main():
    # Wrap the data
    f = gzip.open(ltpath + learning_table_to_use, "rb")
    questions, targets = pickle.load(f)
    questions = questions.reshape(questions.shape[0], np.prod(questions.shape[1:]))
    f.close()

    if simplify_to_binary:
        np.equal(targets, True, targets)

    lt = questions.astype(float), targets.astype(int)

    myData = CData(lt, cross_val=crossval, header=None, pca=pca)
    if reshape:
        assert not pca, "Why would you shape PCA transformed data?"
        myData.data = myData._datacopy = myData.data.reshape(myData.N + myData.n_testing, 1, 60, 60)
    if standardize:
        assert not pca, "Why would you standardize PCA transformed data?"
        myData.standardize()

    # Create network
    net = netclass(myData, eta=eta, lmbd=lmbd, cost=cost)

    # print("Initial test: T", net.evaluate()[1], "L", net.evaluate("learning")[1])
    score = net.train(epochs, batch_size)

    while 1:
        X = np.arange(len(score[0]))
        plt.plot(X, score[0], "b", label="T")
        plt.plot(X, score[1], "r", label="L")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()

        more = int(input("----------\nMore? How much epochs?\n> "))

        if more < 1:
            break

        ns = net.train(int(more), 100)
        score[0].extend(ns[0])
        score[1].extend(ns[1])


def sanity_check():
    from csxnet.datamodel import mnist_to_lt

    print("ATTENTION! This is a sanity test on MNIST data!")

    mnistlt = mnist_to_lt(ltpath+"mnist.pkl.gz")
    mnistdata = CData(mnistlt, cross_val=.1)
    del mnistlt

    net = netclass(mnistdata, eta=0.5, lmbd=5.0, cost=Xent)
    score = net.train(epochs=10, batch_size=10)

    while 1:
        X = np.arange(len(score[0]))
        plt.plot(X, score[0], "b", label="T")
        plt.plot(X, score[1], "r", label="L")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()

        more = int(input("----------\nMore? How much epochs?\n> "))

        if more < 1:
            break

        ns = net.train(int(more), 100)
        score[0].extend(ns[0])
        score[1].extend(ns[1])


learning_table_to_use = "onezero.pkl.gz"

crossval = 0.3
pca = 0
standardize = True
reshape = True
simplify_to_binary = True

hiddens = (300, 100)
drop = 0.0
act_fn_H = Sigmoid
cost = MSE
aepochs = 0
epochs = 50
batch_size = 10
eta = 0.1
eta_decay = 0.0
lmbd = 0.0
netclass = CNNexplicit

if __name__ == '__main__':
    # main()
    sanity_check()
