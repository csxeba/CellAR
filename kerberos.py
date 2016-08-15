from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD

# configuration: hiddens, conv, filters, pool, runs, epochs, batch_size, eta, lmbd1, lmbd2, costfn
# Cconf1 = (300, 75),       3,      3,      2,  10,     30,     10,      0.1,   0.0,   0.0,  "Xent"


def build_CNN(inshape, outshape):
    model = Sequential()
    model.add(Convolution2D(3, 3, 3, input_shape=inshape, activation="tanh"))
    model.add(MaxPooling2D())
    model.add(Dense(300, activation="tanh"))
    model.add(Dense(75, activation="tanh"))
    model.add(Dense(outshape, activation="softmax"))
    model.compile(optimizer=SGD(lr=0.1), loss="categorical_crossentropy")


def load_dataset(crossval, dataset):
    from csxdata.const import roots
    from csxdata.frames import CData
    import pickle
    import gzip

    fl = gzip.open(roots["lt"] + dataset + ".pkl.gz")
    data = CData(pickle.load(fl), cross_val=crossval, header=False, standardize=True)



