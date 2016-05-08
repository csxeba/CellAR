import pickle
import time

import numpy as np
from PIL import Image, ImageTk
from stacksToTiles import pull_image
from experiment import FFNetThinkster, CNNexplicit

# brainpath = "ConvFullTrained.bro"
brainpath = "FCFeedForwardBrain.bro"
tifpath = "/data/Prog/Diploma/ProbaBemenet03.tif"


def inspect_run(blob, net):
    from tkinter import Tk, Label, Frame
    preds = net.predict(np.divide(np.subtract(blob, net.data.mean), net.data.std))

    tk = Tk()
    for i in range(10):
        for j in range(9):
            slc = blob[i*10 + j]
            print(preds[i*10 + j], end="\t")

            pic = Image.fromarray(slc.reshape(60, 60).astype("uint8"), mode="L")
            pic.resize((pic.size[0], pic.size[1]))
            pic = ImageTk.PhotoImage(pic)

            Label(tk, image=pic).grid(row=i, column=j)

        print()

    tk.mainloop()


def tif_to_blobs(array):
    """The supplied array of images gets chopped up.

    60x60 tiles are created and stacked into a massive 3D NumPy array"""
    xsteps, ysteps = array.shape[1] // 60, array.shape[2] // 60
    blobs = []
    coords = []

    for i in range(array.shape[0]):
        slices = []
        crds = []
        for x in range(xsteps):
            startx = x * 60
            endx = startx + 60
            for y in range(ysteps):
                starty = y * 60
                endy = starty + 60
                crds.append((startx, endy))
                slices.append(array[i, startx:endx, starty:endy].reshape(1, 60, 60))
        coords.append(crds)
        blobs.append(np.stack(slices, axis=0))

    for i in range(array.shape[0]):
        slices = []
        for x in range(xsteps-1):
            startx = 30 + (x * 60)
            endx = startx + 60
            for y in range(ysteps-1):
                starty = 30 + (y * 60)
                endy = starty + 60
                slices.append(array[i, startx:endx, starty:endy].reshape(1, 60, 60))
        blobs.append(np.stack(slices, axis=0))

    return blobs


def wake_ai(path):
    netfl = open(path, "rb")
    network = pickle.load(netfl)
    netfl.close()

    print("Artificial Intelligence is awakened.")
    network.describe(1)

    # tacc, lacc = network.evaluate(), network.evaluate("learning")
    # if isinstance(tacc, tuple):
    #     tacc, lacc = tacc[1], lacc[1]
    # print("Network accuracy on T: {} L: {}".format(tacc, lacc))

    return network


def count(blobs, network):
    predictions = []
    for blob in blobs:
        blob -= network.data.mean
        blob /= network.data.std

        prediction = network.predict(blob)
        prediciton = np.sum(prediction)
        predictions.append(prediciton)

    return predictions


def standardize(ar):
    mean = ar.mean(axis=0)
    std = ar.std(axis=0)
    ar -= mean
    ar /= std

def main():
    brain = wake_ai(brainpath)
    tif = pull_image(tifpath, downscale=False)
    standardize(tif)
    blobs = tif_to_blobs(tif)
    # inspect_run(blobs[0], brain)
    answers = count(blobs, brain)
    assert len(answers) == len(blobs)
    logchain = ""
    for i in range(tif.shape[0]):
        logchain += ("{}\t{}".format(i, answers[i]))
        print("Frame number {} contains {} floating cells.".format(i, answers[i]))
    logfl = open("log.txt", "w")
    logfl.write(logchain)
    logfl.close()

if __name__ == '__main__':
    start = time.time()
    main()
    print("Time required: {} s".format(int(time.time()-start)))
