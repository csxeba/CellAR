"""Cell division detection with Artificial Neural Networks"""
import pickle
import gzip

import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from csxnet.datamodel import RData, CData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.brainforge.Utility.cost import Xent, MSE
from csxnet.brainforge.Utility.activations import *
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
        self.predict = theano.function((X), outputs=(prediction,))

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


class CNNetTheano(ConvNet):
    def __init__(self, data, eta, lmbd):
        nfilters = 1
        cfshape = (5, 5)
        pool = 2
        hidden_fc = 120
        ConvNet.__init__(self, data, eta, lmbd, nfilters, cfshape, pool, hidden_fc)

    def train(self, epochs, batch_size):
        scores = [list(), list()]
        for epoch in range(1, epochs+1):
            self.learn(batch_size)
            tcost, tscore = self.evaluate("testing")
            lcost, lscore = self.evaluate("learning")
            scores[0].append(tscore)
            scores[1].append(lscore)
            print("Epoch {} done! Last cost: {}".format(epoch, lcost))
            print("T: {}\tL: {}".format(scores[0][-1], scores[1][-1]))
            if epoch % 10 == 0 and eta_decay > 0:
                self.eta -= eta_decay
                print("ETA DECAYED TO", self.eta)

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


learning_table_to_use = "cssmall.pkl.gz"

crossval = 0.1
pca = 0
standardize = True
reshape = True
simplify_to_binary = True
hiddens = (400, 200)
drop = 0.0
act_fn_H = Sigmoid

cost = MSE
aepochs = 0
epochs = 50
batch_size = 25
eta = 0.5
eta_decay = 0.0
lmbd = 0.0
netclass = CNNetTheano


def main():
    # Wrap the data and build the net from the supplied hyperparameters

    f = gzip.open(ltpath + learning_table_to_use, "rb")
    questions, targets = pickle.load(f)
    questions = questions.reshape(questions.shape[0], np.prod(questions.shape[1:]))
    f.close()

    if simplify_to_binary:
        np.equal(targets, True, targets)

    lt = questions.astype(float), targets.astype(int)

    myData = CData(lt, cross_val=crossval, header=None, pca=pca)
    if reshape:
        myData.data = myData._datacopy = myData.data.reshape(myData.N + myData.n_testing, 1, 60, 60)
    if standardize or not pca:
        myData.standardize()
    net = netclass(myData, eta=eta, lmbd=lmbd)

    print("Initial test: T", net.evaluate(), "L", net.evaluate("learning"))

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


if __name__ == '__main__':
    main()