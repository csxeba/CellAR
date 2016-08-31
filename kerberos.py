import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import *

from csxdata import roots

# configuration: hiddens, conv, filters, pool, runs, epochs, batch_size, eta, lmbd1, lmbd2, costfn
# Cconf1 = (300, 75),       3,      3,      2,  10,     30,     10,      0.1,   0.0,   0.0,  "Xent"
from keras.regularizers import WeightRegularizer

DATADIM = (1, 60, 60)


costfn = SGD(lr=0.01, momentum=0.9, nesterov=True)


class ArchitectureBase(Sequential):

    name = ""

    def __init__(self, name):
        Sequential.__init__(self, name=name)

    def save2(self, path=None):
        if path is None:
            path = roots["brains"]

        import pickle
        import gzip

        with gzip.open(path + self.name + ".ker", "wb") as outfl:
            pickle.dump(self, outfl)
            outfl.close()

    @classmethod
    def load(cls):
        from reload import pull_keras_brain
        return pull_keras_brain(roots["brain"] + cls.name + ".ker")


class LeNet(ArchitectureBase):

    name = "LeNet-like"

    def __init__(self):
        ArchitectureBase.__init__(self, name=LeNet.name)
        print("Building LeNet-like CNN!")
        # (1, 60, 60) = 3 600
        self.add(Convolution2D(7, 11, 11, input_shape=DATADIM, activation="relu"))
        # (3, 50, 50) = 7 500
        self.add(MaxPooling2D())
        # (3, 25, 25) = 1 875
        self.add(Convolution2D(13, 6, 6, activation="relu"))
        # (7, 20, 20)  = 2 800
        self.add(MaxPooling2D())
        # (7, 10, 10)  = 700
        self.add(Flatten())
        self.add(Dense(120, activation="tanh", W_regularizer=WeightRegularizer(0.0, 2.0)))
        # (700x120) = 84 000
        self.add(Dense(2, activation="softmax"))
        # (2x120)   = 240
        self.compile(optimizer=costfn, loss="categorical_crossentropy",
                     metrics=["accuracy"])


class DenseNet(ArchitectureBase):

    name = "DenseNet"

    def __init__(self):
        ArchitectureBase.__init__(self, "DenseNet")
        hiddens = (1200,)

        self.add(Flatten(input_shape=DATADIM))
        for h in hiddens:
            self.add(Dense(h, activation="relu"))
        self.add(Dense(1, activation="sigmoid"))
        self.compile(costfn, loss="binary_crossentropy", metrics=["accuracy"])


class FullyConvolutional(ArchitectureBase):

    name = "FullyConvolutional"

    def __init__(self):
        ArchitectureBase.__init__(self, name=FullyConvolutional.name)
        print("Building fully convolutional network!")
        # (1, 60, 60) = 3600
        self.add(Convolution2D(7, 21, 21, input_shape=DATADIM, activation="relu"))
        # (3, 40, 40) = 4800
        self.add(Convolution2D(7, 21, 21, activation="relu"))
        # (3, 20, 20) = 1200
        self.add(Convolution2D(5, 11, 11, activation="relu"))
        # (5, 10, 10) = 500
        self.add(Convolution2D(7, 6, 6, activation="relu"))
        # (7, 5, 5)  = 175
        self.add(Flatten())
        self.add(Dense(1, activation="relu"))
        self.compile(optimizer=costfn, loss="binary_crossentropy",
                     metrics=["accuracy"])
        # Best score so far: 83% (L) -- 71% (T)


def load_dataset(dataset, preparation, crossval=0.2):
    from csxdata.frames import CData

    preparation = "tiles" if preparation is None else preparation

    data = CData(roots["lt"] + dataset + "_" + preparation + ".pkl.gz",
                 cross_val=crossval, header=False)
    data.indeps = np.greater(data.indeps, 0).astype(int)
    data.categories = list(set(data.indeps))
    data.reset_data()
    data.transformation = "std"
    print("{} categories: {}".format(dataset, data.categories))
    return data


def run(architecture, dataset, preparation=None, rebuild=True):
    data = load_dataset(dataset, preparation)
    inshape, outputs = data.neurons_required
    X, y, validation = data.learning, data.lindeps, (data.testing, data.tindeps)
    print("Loaded data of shape:", inshape)

    net = architecture() if rebuild else from_loaded(architecture)
    net.summary()
    print("Initial cost: {} initial acc: {}".format(*net.evaluate(validation[0], validation[1], verbose=0)))
    # net.fit(X, y, batch_size=20, nb_epoch=30, validation_data=validation, shuffle=True)
    net.fit_generator(data.batchgen(20, infinite=True), data.N*100, nb=10, validation_data=validation)
    net.save2()


def pretrained(architecture, rebuild=True):

    def pretrain_on_xonezero():
        xonezero = load_dataset("xonezero", None)
        pValid = xonezero.table("testing")
        print("Initial cost: {} initial acc: {}".format(*net.evaluate(pValid[0], pValid[1], verbose=0)))
        # net.fit(pX, pY, batch_size=100, nb_epoch=30, validation_data=pValid, shuffle=True)
        net.fit_generator(xonezero.batchgen(100, infinite=True), xonezero.N*5, nb_epoch=2, validation_data=pValid)

    def train_on_big_dataset():
        data = load_dataset("big", None)
        validation = data.table("testing")
        print("Initial cost: {} initial acc: {}"
              .format(*net.evaluate(validation[0], validation[1], verbose=0)))
        print("Percent of zeros in learning: {}%"
              .format(round((1 - data.lindeps.sum() / data.N)*100, 3)))
        print("Percent of zeros in testing: {}%"
              .format(round((1 - data.tindeps.sum() / data.N) * 100, 3)))

        # net.fit(X, y, batch_size=100, nb_epoch=30, validation_data=validation, shuffle=True,
        #         sample_weight=w)
        net.fit_generator(data.batchgen(100, infinite=True), data.N*10, nb_epoch=5, validation_data=validation)

    net = architecture() if rebuild else from_loaded(architecture)
    net.summary()

    pretrain_on_xonezero()
    train_on_big_dataset()

    net.save2()


def from_loaded(architecture, monitor_dataset=None, monitor_data_preparation=None):
    network = architecture.load()
    if monitor_dataset is not None:
        data = load_dataset(monitor_dataset, monitor_data_preparation)
        print("Reloaded net performace monitoring!")
        print("Cost: {}; Acc: {}".format(*network.evaluate(data.testing, data.tindeps, verbose=0)))
    return network


def prediction(architecture):
    network = architecture.load()
    data = load_dataset("big", preparation=None, crossval=0.2)
    X, y = data.learning, data.lindeps
    np.greater_equal(y, 1, out=y)
    where1 = np.argwhere(y)
    where0 = np.argwhere(np.logical_not(y))
    X1 = X[where1.reshape(where1.shape[0])]
    X0 = X[where0.reshape(where0.shape[0])]
    preds1 = network.predict(X1)
    pr1_YES = np.sum(np.greater(preds1[:, 1], 0.5))
    preds0 = network.predict(X0)
    pr0_NO = np.sum(np.greater(preds0[:, 0], 0.5))

    print("ALL: {};\nYES: {}; RATE: {}\nNO:  {}; RATE: {}"
          .format(preds1.shape[0] + preds0.shape[0],
                  pr1_YES, pr1_YES / preds1.shape[0],
                  pr0_NO, pr0_NO / preds0.shape[0]))


if __name__ == '__main__':
    pretrained(LeNet)
    print("Finite Incantatum!")
