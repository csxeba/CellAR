"""Cell division detection with Artificial Neural Networks"""
import sys
import time
import os

from csxnet.datamodel import CData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.brainforge.Utility.cost import Xent, MSE
from csxnet.brainforge.Utility.activations import *
from csxnet.thNets.thANN import ConvNetExplicit, ConvNetDynamic

dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"
ltroot = dataroot + "learning_tables/"
brainroot = dataroot + "brains/"


class CNNexplicit(ConvNetExplicit):
    def __init__(self, data, rate, l1, l2, momentum, costfn):
        nfilters = 2
        cfshape = (3, 3)
        pool = 2
        hidden1, hidden2 = hiddens[0], hiddens[1]
        if not isinstance(costfn, str):
            costfn = str(costfn())
        costfn = "MSE" if costfn is MSE else "Xent"
        ConvNetExplicit.__init__(self, data, rate, l1, l2,
                                 nfilters, cfshape, pool,
                                 hidden1, hidden2,
                                 costfn)

    def train(self, eps, bsize):
        if bsize == "full":
            print("ATTENTION! Learning in full batch mode! m =", self.data.N)
            bsize = self.data.N
        scores = [list(), list()]
        for epoch in range(1, eps+1):
            self.learn(bsize)
            if epoch % 1 == 0:
                tcost, tscore = self.evaluate("testing")
                lcost, lscore = self.evaluate("learning")
                scores[0].append(tscore)
                scores[1].append(lscore)
                print("Epoch {}/{} done! Cost: {}".format(epoch, eps, lcost))
                print("T: {}\tL: {}".format(scores[0][-1], scores[1][-1]))

        return scores

    def save(self, path):
        import pickle
        outfl = open(path, "wb")
        pickle.dump(self, outfl)
        outfl.close()


class CNNdynamic(ConvNetDynamic):
    def __init__(self, data, l_rate, l1, l2, momentum, costfn):
        if data.pca:
            raise RuntimeError("CNN received PCAd data...")

        ConvNetDynamic.__init__(self, data, l_rate, l1, l2, momentum, costfn)

        self.add_convpool(conv=3, filters=2, pool=2)
        for hid in hiddens:
            if isinstance(hid, str):
                hid = int(hid[:-1])
                print("DropoutLayer not yet implemented! Falling back to FCLayer!")
            self.add_fc(hid, act_fn_H)

        self.finalize()

    def train(self, eps, bsize):
        if bsize == "full":
            print("ATTENTION! Learning in full batch mode! m =", self.data.N)
            bsize = self.data.N
        scores = [list(), list()]
        for epoch in range(1, eps + 1):
            self.learn(bsize)
            if epoch % 1 == 0:
                tcost, tscore = self.evaluate("testing")
                lcost, lscore = self.evaluate("learning")
                scores[0].append(tscore)
                scores[1].append(lscore)
                print("Epoch {}/{} done! Cost: {}".format(epoch, eps, lcost))
                print("T: {}\tL: {}".format(scores[0][-1], scores[1][-1]))

        return scores


class FFNetThinkster(Network):
    def __init__(self, data, lrate, l1, l2, momentum, costfn):
        if isinstance(costfn, str):
            costfn = {"mse": MSE, "xent": Xent}[costfn.lower()]
        else:
            costfn = cost
        if isinstance(act_fn_H, str):
            actfn = {"sig": Sigmoid, "tan": Tanh,
                     "rel": ReL, "lin": Linear}[act_fn_H[:3]]
        else:
            actfn = act_fn_H

        Network.__init__(self, data, lrate, l1, l2, momentum, costfn)

        for h in hiddens:
            if isinstance(h, int) and h:
                self.add_fc(h, activation=actfn)
            if isinstance(h, str) and h:
                h = int(h[:-1])
                self.add_drop(h, drop, activation=ReL)
        self.finalize_architecture()

    def train(self, eps, bsize):
        scores = [list(), list()]
        for epoch in range(1, eps+1):

            self.learn(batch_size=bsize)

            if epoch % 1 == 0:
                scores[0].append(self.evaluate())
                scores[1].append(self.evaluate("learning"))
                print("Epoch {}/{} done! Err: {}".format(epoch, eps, self.error))
                print("Acc:", scores[0][-1], scores[1][-1])
            if epoch == eps:
                self.evaluate()

        return scores


def run(lt):
    print("Wrapping learning data...")
    myData = wrap_data(ltroot + lt)

    print("Building network...")
    net = network_class(myData, eta, lmbd1, lmbd2, mu, cost)

    print("Initial score:", net.evaluate())

    net.describe(1)

    score = net.train(epochs, batch_size)

    while 1:
        net.describe()
        display(score)

        more = int(input("----------\nMore? How much epochs?\n> "))
        try:
            more = int(more)
        except ValueError:
            print("Now you killed my script. Thanks.")
            more = 0
        if more < 1:
            break

        ns = net.train(more, batch_size)
        score[0].extend(ns[0])
        score[1].extend(ns[1])

    return net


def run2(path_to_lt, runs=5):
    scores = []
    for r in range(1, runs+1):
        start = time.time()

        print("Wrapping learning data...")
        myData = wrap_data(path_to_lt)

        print("Building network...")
        net = network_class(myData, eta, lmbd1, lmbd2, mu, cost)

        net.train(epochs, batch_size)
        scores.append(net.evaluate())
        print("Run {} took {} seconds".format(r, int(time.time()-start)))

    print("SCORES:")
    print(scores)
    return scores


def longrun(runs=10):
    lts = "/data/Prog/data/lts/"
    logchain = ""
    for lt in os.listdir(lts):
        print("\nDATASET:", lt)
        logchain += "----------\nDATASET: {}\n".format(lt)
        for r in range(1, 1+runs):
            runchain = ""
            start = time.time()
            myData = wrap_data(lts + lt)
            net = FFNetThinkster(myData, eta, 0.0, 0.0, 0.0, "Xent")
            runchain += "RUN {} INITIAL SCORE: {}\n".format(r, str(net.evaluate()))
            net.train(epochs, batch_size)
            runchain += "RUN {}  FINAL  SCORE: {}\n".format(r, str(net.evaluate()))
            runchain += "Time: {} s\n".format(int(time.time() - start))
            print("_____\n" + runchain + "______")
            logchain += runchain
    print("Finished run!")
    logfl = open("log.txt", "w")
    logfl.write(logchain)
    logfl.close()


def sanity_check():
    from csxnet.datamodel import mnist_to_lt

    print("ATTENTION! This is a sanity test on MNIST data!")

    lrate = 0.5
    l1 = 2.0
    l2 = 0.0
    momentum = 0.9
    costfn = "xent"

    no_epochs = 10
    bsize = 10

    print("Pulling data...")
    mnistlt = mnist_to_lt(ltroot + "mnist.pkl.gz")
    mnistdata = CData(mnistlt, cross_val=.1)
    del mnistlt

    print("Building Neural Network...")
    net = network_class(mnistdata, lrate, l1, l2, momentum, costfn)
    net.describe(1)
    print("Training Neural Network...")
    score = net.train(eps=no_epochs, bsize=bsize)

    while 1:
        net.describe()
        display(score)
        more = int(input("More? How much epochs?\n> "))

        if more < 1:
            break

        ns = net.train(int(more), bsize)
        score[0].extend(ns[0])
        score[1].extend(ns[1])


def wrap_data(path_to_lt):
    import pickle
    import gzip

    f = gzip.open(path_to_lt, "rb")
    questions, targets = pickle.load(f)
    questions = questions.reshape(questions.shape[0], np.prod(questions.shape[1:]))
    f.close()

    if simplify_to_binary:
        np.equal(targets, True, targets)

    lt = questions.astype(float), targets.astype(int)

    myData = CData(lt, cross_val=crossval, header=None, pca=pca)
    if reshape:
        assert not pca, "Why would you reshape PCA transformed data?"
        myData.data = myData.data.reshape(myData.N + myData.n_testing, 1, 60, 60)
        myData.split_data()
    if standardize:
        assert not pca, "Why would you standardize PCA transformed data?"
        myData.standardize()

    return myData


def display(score):
    import matplotlib.pyplot as plt
    X = np.arange(1, len(score[0])+1)
    plt.plot(X, score[0], "b", label="T")
    plt.plot(X, score[1], "r", label="L")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.axis([1, X.max(), 0.0, 1.0])
    plt.show()


def savebrain(brain, flname="autosave.bro"):
    import pickle
    outfl = open(flname, "wb")
    pickle.dump(brain, outfl)
    outfl.close()

network_class = FFNetThinkster
learning_table = "cssmall.pkl.gz"

# Paramters for the data wrapper
crossval = 0.3
pca = 0
standardize = True
reshape = True
simplify_to_binary = True

# Parameters for the neural network
hiddens = (300, 180, 60)  # string entry of format "60d" means a dropout layer with 60 neurons
aepochs = 0  # Autoencode for this many epochs
epochs = 30
drop = 0.5  # Chance of dropout (if there are droplayers)
batch_size = 10
eta = 0.03
lmbd1 = 0.0
lmbd2 = 0.0
mu = 0.0
act_fn_H = "tanh"  # Activation function of hidden layers
cost = "Xent"  # MSE / Xent cost functions supported


if __name__ == '__main__':
    # sanity_check()
    network = run(learning_table)
    network.save(brainroot + "autosave.bro")
    # longrun(5)
    # run2(5)
