import numpy as np

from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import *

from csxdata.const import roots

# configuration: hiddens, conv, filters, pool, runs, epochs, batch_size, eta, lmbd1, lmbd2, costfn
# Cconf1 = (300, 75),       3,      3,      2,  10,     30,     10,      0.1,   0.0,   0.0,  "Xent"

DATADIM = (1, 60, 60)


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
        self.add(Dense(120, activation="sigmoid"))
        # (700x120) = 84 000
        self.add(Dense(1, activation="sigmoid"))
        # (1x120)   = 120
        self.compile(optimizer=Adagrad(), loss="binary_crossentropy",
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
        self.compile(Adagrad(), loss="binary_crossentropy", metrics=["accuracy"])


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
        self.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss="binary_crossentropy",
                     metrics=["accuracy"])
        # Best score so far: 83% (L) -- 71% (T)


def load_dataset(dataset, preparation, crossval=0.2):
    from csxdata.frames import CData
    import pickle
    import gzip

    if preparation is None:
        preparation = "tiles"

    fl = gzip.open(roots["lt"] + dataset + "_" + preparation + ".pkl.gz")
    data = CData(pickle.load(fl), cross_val=crossval, header=False, standardize=True)
    return data


def run(architecture, dataset, preparation=None):
    if preparation is None:
        preparation = "tiles"
    data = load_dataset(dataset, preparation)
    inshape, outputs = data.neurons_required
    X, y, validation = data.learning, data.lindeps, (data.testing, data.tindeps)
    print("Loaded data of shape:", inshape)

    net = architecture()
    net.summary()
    print("Initial cost: {} initial acc: {}".format(*net.evaluate(validation[0], validation[1], verbose=0)))
    net.fit(X, y, batch_size=20, nb_epoch=30, validation_data=validation, shuffle=True)
    net.save2()


def from_loaded(architecture, dataset, preparation):
    network = architecture.load()
    data = load_dataset(dataset, preparation)
    print("Reloaded net performace monitoring!")
    print("Cost: {}; Acc: {}".format(*network.evaluate(data.testing, data.tindeps, verbose=0)))
    return network


def prediction(architecture):
    network = architecture.load()
    data = load_dataset("big", preparation=None, crossval=0.2)
    X, y = data.learning, data.lindeps
    np.greater_equal(y, 1, out=y)
    where1 = np.argwhere(y)
    X1 = X[where1.reshape(where1.shape[0])]
    preds = network.predict(X1)

    pass

if __name__ == '__main__':
    run(DenseNet, "xonezero", None)
