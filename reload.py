import pickle

from csxdata import roots

brainroot = roots["brain"]
brainpath = "76CNNexplicit.bro"

ltroot = roots["lt"]
ltpath = "onezero.pkl"


def pull_brain(path):
    fl = open(path, "rb")
    brain = pickle.load(fl)
    fl.close()
    brain.describe(1)
    print("Score on testing:", brain.evaluate("testing")[1])
    return brain


def pull_keras_brain(path):
    import gzip
    with gzip.open(path, "rb") as fl:
        brain = pickle.load(fl)
        fl.close()
    print(brain.name, "has been awakened!")
    brain.summary()
    return brain


if __name__ == '__main__':
    brian = pull_brain(brainroot+brainpath)
