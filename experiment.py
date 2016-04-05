"""Cell division detection with Artificial Neural Networks"""
import numpy as np
import theano
import theano.tensor as T

from csxnet.datamodel import CData
from csxnet.brainforge.Architecture.NNModel import Network
from csxnet.brainforge.Utility.cost import Xent, MSE

ltpath = "/data/Prog/data/learning_tables/"


class FFNetTheano:
    def __init__(self, data, eta, lmbd):
        """Simple Feed Forward architecture implemented with Theano"""
        self.data = data
        self.data.flatten()

        inputs = 3600
        hiddens = 1000

        l2term = 1 - ((eta * lmbd) / data.N)

        outputs = len(data.categories)

        X = T.matrix("X")  # Batch of inputs
        Y = T.matrix("Y")  # Batch of targets

        m = T.scalar()  # Input batch size

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
        for epoch in range(epochs):
            for no, batch in enumerate(self.data.batchgen(batch_size)):
                self._fit(batch[0], batch[1], batch_size)
                if no % 10 == 0 and no != 0:
                    cost, acc = self.evaluate()
                    print("Batch", no)
                    print("Cost:", cost)
                    print("AccT:", acc)
            print("---- Epoch {}/{} Done! ----".format(epoch, epochs))

    def evaluate(self, on="testing"):
        cost, answers = self.predict(self.data.table(on))
        rate = np.sum(np.equal(answers, self.data.tindeps))
        return cost, rate


class FFNetThinkster(Network):
    def __init__(self, data, eta, lmbd):
        Network.__init__(self, data, eta, lmbd, MSE)
        self.add_fc(1000)
        self.finalize_architecture()

    def train(self, epochs, batch_size):
        scores = [list(), list()]
        for epoch in range(epochs):
            self.learn(batch_size=batch_size)
            scores[0].append(self.evaluate())
            scores[1].append(self.evaluate("learning"))
            print("Epoch {}".format(epoch))
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
    import pickle, gzip
    import matplotlib.pyplot as plt

    f = gzip.open(ltpath + "gorlt.pkl.gz", "rb")
    questions, targets = pickle.load(f)
    f.close()

    np.divide(questions, 255, out=questions)
    lt = questions, targets

    myData = CData(lt, cross_val=0.1, pca=0)
    net = FFNetThinkster(myData, eta=0.5, lmbd=2.0)
    print("Initial test:", net.evaluate())
    score = net.train(10, 50)

    X = np.arange(len(score[0]))
    plt.plot(X, score[0], "b", label="T")
    plt.plot(X, score[1], "r", label="L")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


