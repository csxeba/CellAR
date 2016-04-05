import random

import numpy as np

from mylibs.utilities import ravel_to_matrix as rtm


YAY = 1.0
NAY = 0.0


class _Data:
    """Base class for Data Wrappers"""
    def __init__(self, source, cross_val, indeps_n, header, sep, end):
        if isinstance(source, np.ndarray):
            headers, data, indeps = parsearray(source, header, indeps_n)
        elif "mnist" in source:
            headers, data, indeps = parseMNIST(source)
        elif isinstance(source, tuple):
            headers, data, indeps = parseLT(source)
        else:
            headers, data, indeps = parsefile(source, header, indeps_n, sep, end)

        n_testing = int(len(data) * (1 - cross_val))

        data, indeps = shuffle((data, indeps))

        self.headers = headers
        self.data = data
        self.indeps = indeps  # independent variables

        self.learning = data[:n_testing]
        self.lindeps = indeps[:n_testing]
        self.testing = data[n_testing:]
        self.tindeps = indeps[n_testing:]

        self.N = self.learning.shape[0]

    def table(self, data):
        """Returns a learning table"""
        dat = {"l": self.learning,
               "t": self.testing}[data[0]]
        dep = {"l": self.lindeps,
               "t": self.tindeps}[data[0]]

        # I wasn't sure if the order gets messed up or not, but it seems it isn't
        # Might be wise to implement thisas a test method, because shuffling
        # transposed data might lead to surprises
        return dat, dep

    def batchgen(self, bsize, data="learning"):
        tab = self.table(data)
        tsize = len(tab[0])
        start = 0
        end = start + bsize

        while start < tsize:
            if end > tsize:
                end = tsize

            out = (tab[0][start:end], tab[1][start:end])

            start += bsize
            end += bsize

            yield out


class CData(_Data):
    """
    This class is for holding categorical learning data. The data is read
    from the supplied source .csv semicolon-separated file. The file should
    contain a table of numbers with headers for columns.
    The elements must be of type integer or float.
    """

    def __init__(self, source, cross_val=.2, header=True, sep="\t", end="\n", flatten=False):
        _Data.__init__(self, source, cross_val, 1, header, sep, end)

        # In categorical data, there is only 1 independent categorical variable
        # which is stored in a 1-tuple or 1 long vector. We free it from its misery
        if isinstance(self.indeps[0], tuple):
            self.indeps = np.array([d[0] for d in self.indeps])
            self.lindeps = np.array([d[0] for d in self.lindeps])
            self.tindeps = np.array([d[0] for d in self.tindeps])

        # We extract the set of categories. set() removes duplicate values.
        self.categories = list(set(self.indeps))

        # Every category gets associated with a y long array, where y is the
        # number of categories. Every array gets filled with zeros.
        targets = np.zeros((len(self.categories),
                            len(self.categories)))

        # The respective element of the array, corresponding to the associated
        # category is set to 1.0. Thus if 10 categories are available, then category
        # No. 3 is represented by the following array: 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
        np.add(targets, NAY, out=targets)
        np.fill_diagonal(targets, YAY)

        # A dictionary is created, which associates the category names to the target
        # arrays representing them.
        self.dictionary = {category: target for category, target in
                           zip(self.categories, targets)}

        if flatten:
            self.flatten()

    def table(self, data="learning"):
        """Returns a learning table"""
        datum, indep = _Data.table(self, data)
        adep = np.array([self.dictionary[de] for de in indep])
        datum, indep = shuffle((datum, indep))
        return datum, adep

    def translate(self, output):
        """Translates a Brain's output to a human-readable answer"""
        output = output.ravel().tolist()
        answer = output.index(max(output))
        return self.categories[answer]

    def dummycode(self, data="testing"):
        d = self.tindeps if data == "testing" else self.lindeps[:len(self.tindeps)]
        return np.array([self.categories.index(x) for x in d])

    def neurons_required(self):
        """Calculates the required number of input and output neurons
         to process this data. Shape is not calculated (yet)."""
        print("Warning! CData.neurons_required is OBSOLETE!")
        return len(self.data[0]), len(self.categories)

    def test(self, brain, data="testing"):
        """Test a brain against the data.
        Returns the rate of rights answers"""
        print("Warning! CData.test() is OBSOLETE!")
        ttable = self.table(data)
        thoughts = brain.think(ttable[0])
        answers = [self.translate(thought) for thought in thoughts]
        targets = [self.translate(target) for target in ttable[1]]
        results = [int(x == y) for x, y in zip(answers, targets)]
        return sum(results) / len(results)

    def flatten(self):
        self.data = rtm(self.data)
        self.learning = rtm(self.data)
        self.testing = rtm(self.data)


class RData(_Data):
    def __init__(self, source, cross_val, indeps_n, header, sep=";", end="\n"):
        """
        Class for holding regression learning data. The data is read from the
        supplied source .csv semicolon-separated file. The file should contain
        a table of numbers with headers for columns.
        The elements must be of type integer or float.
        """
        _Data.__init__(self, source, cross_val, indeps_n, header, sep, end)
        self.deps = self.deps.astype(np.float64)
        self.lindeps = self.lindeps.astype(np.float64)
        self.tindeps = self.tindeps.astype(np.float64)

    def neurons_required(self):
        return len(self.data[0]), len(self.deps[0])

    def test(self, brain, data="testing", queue=None):
        ttable = self.table(data)
        thoughts = brain.think(ttable[0])

        assert ttable[1].shape == thoughts.shape, "Targets' shape differ from brain output!"

        if not queue:
            return np.average(np.sqrt(np.square(ttable[1] - thoughts)))
        queue.put(brain)


def parsefile(source, header, deps_n, sep, end):
    file = open(source)  # open mind! :D
    text = file.read()
    text.replace(",", ".")
    file.close()

    lines = text.split(end)

    if header:
        headers = lines[0]
        headers = headers.split(sep)
        lines = lines[1:-1]
    else:
        lines = lines[:-1]
        headers = None

    random.shuffle(lines)
    lines = [line.split(sep) for line in lines]

    deps = np.array([line[:deps_n] for line in lines], dtype=np.string_)
    data = np.array([line[deps_n:] for line in lines], dtype=np.float64)

    return headers, data, deps


def parsearray(X, header, deps_n):
    headers = X[0] if header else None
    matrix = X[1:] if header else X
    deps = matrix[:, :deps_n]
    data = matrix[:, deps_n:]
    return headers, data, deps


def parseMNIST(source):
    import pickle
    import gzip
    f = gzip.open(source, mode="rb")
    # This hack is needed because I'm a lazy ....
    with f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        tr = u.load()
    f.close()
    headers = None
    data = np.concatenate(((tr[0][0]), (tr[1][0]), (tr[2][0])), axis=0).reshape((70000, 1, 28, 28))
    deps = np.concatenate((tr[0][1], tr[1][1], tr[2][1]))

    return headers, data, deps


def parseLT(source):
    return None, source[0], source[1]


def shuffle(learning_table):
    """Shuffles and recreates the learning table"""
    indices = np.arange(learning_table[0].shape[0])
    np.random.shuffle(indices)
    return learning_table[0][indices], learning_table[1][indices]
