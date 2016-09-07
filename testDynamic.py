import sys

import numpy as np

from csxnet.thNets.thANN import ConvNetDynamic
from csxnet.datamodel import CData


dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"
ltroot = dataroot + "/learning_tables/"


def getnet(data, eta, lmbd1, lmbd2, mu, cost):
    network = ConvNetDynamic(data, eta, lmbd1, lmbd2, mu, cost)
    network.add_convpool(3, 2, 2)
    network.add_fc(300, "tanh")
    network.finalize()
    network.describe(1)
    return network


def wrap_data(path, crossval, pca, standardize, reshape, simplify):
    import pickle
    import gzip

    if not pca:
        standardize = True
        reshape = True
    else:
        print("Warning! CNN can't handle PCA transformed data!")
        if standardize:
            print("Warning! Both PCA and standardization was requested. Ignoring latter!")
            standardize = False
        if reshape:
            print("Warning! Both PCA and reshaping was requested. Ignoring latter!")
            reshape = False

    questions, targets = pickle.load(gzip.open(path))

    if reshape:
        questions = np.reshape(questions, (questions.shape[0], 1, 60, 60))
    if simplify:
        targets = np.greater_equal(targets, 1).astype("int32")

    data = CData((questions, targets), cross_val=crossval, pca=pca)

    if standardize:
        data.standardize()

    return data


def teach(net: ConvNetDynamic, eps: int, bsize: int, evaluate=False):
    scores = []
    for epoch in range(1, eps+1):
        net.learn(bsize)
        tcost, tacc = net.evaluate("testing")
        lcost, lacc = net.evaluate("learning")
        scores.append((tacc, lacc))
        print("Done epoch {}. Cost: {}".format(epoch, lcost))
        print("Acc on T: {}".format(tacc))
        print("Acc on L: {}".format(lacc))
    return net


ltname = "onezero.pkl.gz"

# crossval, pca, stdize, rshp, simplify
dargs = 0.2, 0, True, True, True
# eta, l1, l2, mu, costfn
cargs = 0.1, 0.0, 0.0, 0.0, "Xent"

epochs = 30
batch_size = 10

myData = wrap_data(ltroot+ltname, *dargs)
myNetwork = getnet(myData, *cargs)

teach(myNetwork, epochs, batch_size, evaluate=True)
