"""Cell division detection with Artificial Neural Networks"""
import sys
import time
import datetime

from csxnet.datamodel import CData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.brainforge.Utility.cost import Xent, MSE
from csxnet.brainforge.Utility.activations import *
from csxnet.thNets.thANN import ConvNetExplicit


dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"
ltroot = dataroot + "lts/"
brainroot = dataroot + "brains/"


# Paramters for the data wrapper
crossval = 0.3
standardize = True
reshape = True
simplify_to_binary = True


class CNNexplicit(ConvNetExplicit):
    def __init__(self, data, hiddens, conv, filters, pool, rate, l1, l2, momentum, costfn):
        cfshape = (conv, conv)
        hidden1, hidden2 = hiddens
        if not isinstance(costfn, str):
            costfn = str(costfn())
        costfn = "MSE" if costfn is MSE else "Xent"
        ConvNetExplicit.__init__(self, data, rate, l1, l2,
                                 filters, cfshape, pool,
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


class FFNetThinkster(Network):
    def __init__(self, data, hiddens, lrate, l1, l2, momentum, act_fn_H, costfn):
        if isinstance(costfn, str):
            costfn = {"mse": MSE, "xent": Xent}[costfn.lower()]
        else:
            costfn = costfn
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
                self.add_drop(h, drop, activation=actfn)
        self.finalize_architecture()

    def train(self, eps, bsize):
        scores = [list(), list()]
        for epoch in range(1, eps+1):

            self.learn(batch_size=bsize)

            if epoch % 1 == 0:
                scores[0].append(self.evaluate())
                scores[1].append(self.evaluate("learning"))
                print("Epoch {}/{} done! Cost: {}".format(epoch, eps, self.error))
                print("Acc:", scores[0][-1], scores[1][-1])
            if epoch == eps:
                self.evaluate()

        return scores


def Frun(lt, hiddens, pca, runs, epochs, batch_size, eta, lmbd1, lmbd2, mu, actfn, costfn, architecture):
    print("Wrapping learning data...")
    myData = wrap_data(ltroot + lt, pca)
    print("Testing pictures:", myData.n_testing)

    print("Building network...")
    net = network_class(myData, hiddens, eta, lmbd1, lmbd2, mu, actfn, costfn)

    print("Initial score:", net.evaluate())

    net.describe(1)

    score = net.train(epochs, batch_size)

    while 1:
        net.describe()
        display(score)

        more = input("----------\nMore? How much epochs?\n> ")
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


def Crun(lt, hiddens, conv, filters, pool, runs, epochs, batch_size, eta, lmbd1, lmbd2, costfn):
    print("Wrapping learning data...")
    myData = wrap_data(ltroot + lt, pca=0)
    print("Testing pictures:", myData.n_testing)

    print("Building network...")
    net = CNNexplicit(myData, hiddens, conv, filters, pool, eta, lmbd1, lmbd2, 0.0, costfn)

    print("Initial score:", net.evaluate())

    net.describe(1)

    score = net.train(epochs, batch_size)

    while 1:
        net.describe()
        display(score)

        more = input("----------\nMore? How much epochs?\n> ")
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


def FCconfiguration(args):
    assert len(args) == 12
    hiddens, pca, runs, epochs, batch_size, eta, lmbd1, lmbd2, mu, \
    actfn, costfn, architecture = args
    scores = []
    now = datetime.datetime.now()
    logchain = "\n" + now.strftime("%Y.%m.%d. %H:%M") + "\n"
    logchain += "-*-*-*- CONFIGURATION -*-*-*-\n"
    logchain += "hiddens, pca, runs, epochs, batch_size, eta, lmbd1, lmbd2, mu, actfn, costfn, architecture\n"
    logchain += str(args) + "\n"
    logchain += "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n"

    globstart = time.time()

    for r in range(1, runs + 1):
        print("\nRun {}/{}".format(r, runs))
        logchain += "----------\nRun {}/{}\n".format(r, runs)
        start = time.time()

        myData = wrap_data(ltroot + learning_table)
        net = architecture(myData, hiddens, eta, lmbd1, lmbd2, mu, actfn, costfn)
        net.describe(1)

        net.train(epochs, batch_size)
        scores.append(net.evaluate())
        print("----- Run {} took {} seconds".format(r, int(time.time() - start)))
        logchain += "ACCT: {}\nACCL: {}\n".format(net.evaluate(), net.evaluate("learning"))
        logchain += "----------\n"

    print("SCORES:", str(scores))
    print("SCORES AVG: {}, STD: {}".format(np.mean(scores), np.std(scores)))
    print("*******************")
    logchain += "*** FINISHED RUNNING CONFIGURATION! ***\n"
    logchain += "SCORES: " + str(scores) + "\n"
    logchain += "SCORES AVG: {}, STD: {}\n".format(np.mean(scores), np.std(scores))
    logchain += "TIME   AVG: {}\n".format(int((time.time()-globstart) / runs))
    logchain += "***************************************\n"

    logfl = open("logs/log.txt", "a")
    logfl.write(logchain)
    logfl.close()

    return logchain


def Cconfiguration(args):
    assert len(args) == 11
    hiddens, conv, filters, pool, runs, epochs, batch_size, eta, lmbd1, lmbd2, costfn = args
    scores = []
    now = datetime.datetime.now()
    logchain = "\n" + now.strftime("%Y.%m.%d. %H:%M") + "\n"
    logchain += "-*-*-*- CONFIGURATION -*-*-*-\n"
    logchain += "hiddens, conv, filters, pool, runs, epochs, batch_size, eta, lmbd1, lmbd2, costfn\n"
    logchain += str(args) + "\n"
    logchain += "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n"

    globstart = time.time()

    for r in range(1, runs + 1):
        print("\nRun {}/{}".format(r, runs))
        logchain += "----------\nRun {}/{}\n".format(r, runs)
        start = time.time()

        myData = wrap_data(ltroot + learning_table)
        net = CNNexplicit(myData, hiddens, conv, pool, filters, eta, lmbd1, lmbd2, 0.0, costfn)
        net.describe(1)

        net.train(epochs, batch_size)
        scores.append(net.evaluate()[1])
        print("----- Run {} took {} seconds".format(r, int(time.time() - start)))
        logchain += "ACCT: {}\nACCL: {}\n".format(net.evaluate()[1], net.evaluate("learning")[1])
        logchain += "----------\n"

    print("SCORES:", str(scores))
    print("SCORES AVG: {}, STD: {}".format(np.mean(scores), np.std(scores)))
    print("*******************")
    logchain += "*** FINISHED RUNNING CONFIGURATION! ***\n"
    logchain += "SCORES: " + str(scores) + "\n"
    logchain += "SCORES AVG: {}, STD: {}\n".format(np.mean(scores), np.std(scores))
    logchain += "TIME   AVG: {}\n".format(int((time.time()-globstart) / runs))
    logchain += "***************************************\n"

    logfl = open("logs/log.txt", "a")
    logfl.write(logchain)
    logfl.close()

    return logchain


def final_training(cargs, fargs):

    print("Starting final training!")
    print("Wrapping data...")
    data = wrap_data(ltroot + learning_table, pca=0, cv=0)
    hiddens, conv, filters, pool, runs, epochs, batch_size, eta, lmbd1, lmbd2, costfn = cargs

    print("Creating and training ConvNet...")
    cnet = CNNexplicit(data, hiddens, conv, filters, pool, eta, lmbd1, lmbd2, 0.0, costfn)
    cnet.train(epochs, batch_size)
    cnet.save("ConvFullTrained.bro")

    hiddens, pca, runs, epochs, batch_size, eta, lmbd1, lmbd2, mu, actfn, costfn, architecture = fargs

    print("Creating and training FFNet...")
    fnet = FFNetThinkster(data, hiddens, eta, lmbd1, lmbd2, mu, actfn, costfn)
    fnet.train(epochs, batch_size)
    fnet.save("FFNetFullTrained.bro")

    print("Finished!")


def wrap_data(path_to_lt, pca, cv=crossval):
    import pickle
    import gzip

    f = gzip.open(path_to_lt, "rb")
    questions, targets = pickle.load(f)
    questions = questions.reshape(questions.shape[0], np.prod(questions.shape[1:]))
    f.close()

    if simplify_to_binary:
        np.greater_equal(targets, 1, out=targets)

    lt = questions.astype(float), targets.astype(int)

    myData = CData(lt, cross_val=cv, header=None, pca=pca)
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
    plt.plot(X, score[0], "b", label="Teszt adatsor")
    plt.plot(X, score[1], "r", label="Tanuló adatsor")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.axis([X.min(), X.max(), 0.5, 1.0])
    plt.xlabel("Tanulókorszakok száma")
    plt.ylabel("Helyes predikciók aránya")

    plt.show()


def savebrain(brain, flname="autosave.bro"):
    import pickle
    outfl = open(flname, "wb")
    pickle.dump(brain, outfl)
    outfl.close()


def configuration(*confs):
    for i, args in enumerate(confs):
        print("\n*** CONFIG {} ***".format(i+1))
        if args[-1] is FFNetThinkster:
            FCconfiguration(args)
        else:
            Cconfiguration(args)

network_class = FFNetThinkster
learning_table = "xonezero_bgs.pkl.gz"

# Parameters for the neural network
drop = 0.5  # Chance of dropout (if there are droplayers)

# configuration: hiddens, pca, runs, epochs, batch_size, eta, lmbd1, lmbd2, mu, actfn, costfn, architecture
# Fconf0 = (300, ), 0, 10, 20, 10, 0.03, 0.0, 0.0, 0.0, "tanh", "Xent", FFNetThinkster
# configuration: hiddens, conv, filters, pool, runs, epochs, batch_size, eta, lmbd1, lmbd2, costfn
# Cconf0 = (150, 75), 5, 2, 2, 5, 200, 20, 0.01, 0.0, 0.0, "Xent"

Fconf1 = (300, ), 0, 40, 20, 5, 0.03, 0.0, 0.0, 0.0, "tanh", "Xent", FFNetThinkster
Cconf1 = (300, 75), 3, 3, 2, 10, 30, 10, 0.1, 0.0, 0.0, "Xent"

if __name__ == '__main__':
    Fbrain = Frun(learning_table, *Fconf1)
    Cbrain = Crun(learning_table, *Cconf1)
    # savebrain(Fbrain, "FCFeedForwardBrain.bro")
    # savebrain(Cbrain, "ConvFeedForwardBrain.bro")
