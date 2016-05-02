import pickle
import sys

dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"

brainroot = "brains/"
brainpath = "76CNNexplicit.bro"

ltroot = "learning_tables/"
ltpath = "onezero.pkl"


def pull_brain(path):
    fl = open(path, "rb")
    brain = pickle.load(fl)
    fl.close()
    brain.describe(1)
    print("Score on testing:", brain.evaluate("testing")[1])
    return brain

if __name__ == '__main__':
    brian = pull_brain(dataroot+brainroot+brainpath)
