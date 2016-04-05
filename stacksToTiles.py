import PIL.Image as Img
import numpy as np

TIF_DEPTH = 57

fullpath = "./data/660x480x57x8bit.tif"
midpath = "./data/600x420x57x8bit_mid.tif"
outpath = ".data/tiles/"

# Pulling a TIFF imagestack from HDD and slicing it up


def pull_image(path: str, y: int, x: int):
    """The image is opened and converted to a 3D NumPy array

    The array is returned after feature scaling."""
    out = np.zeros((TIF_DEPTH, y, x), dtype=np.float32)
    fimg = Img.open(path)

    for i in range(TIF_DEPTH):
        fimg.seek(i)
        out[i] = np.array(fimg)

    return out / 255


def slice_up(array):
    """The supplied array of images gets chopped up.

    60x60 tiles are created and stacked into a massive 3D NumPy array"""
    xsteps, ysteps = array.shape[1] // 60, array.shape[2] // 60
    slices = []

    for x in range(xsteps):
        startx = x * 60
        endx = startx + 60
        for y in range(ysteps):
            starty = y * 60
            endy = starty + 60
            slices.append(array[..., startx:endx, starty:endy])

    return np.concatenate(slices)


def img_to_slices(inpath):
    """This method coordinates the Img-slices conversion"""
    this = "mid" if "mid" in inpath else "full"
    shape = (420, 600) if this is "mid" else (480, 660)
    print("Pulling", this)
    array = pull_image(inpath, *shape)
    print("Slicing", this)
    slices = slice_up(array)
    return slices


def write_to_files(array, index, batch_no, queue):
    """Writes tiles to files from supplied array to PNG images"""
    array = np.multiply(array, 255, out=array)
    prefix = "0"
    suffix = ".png"
    print("Proc", batch_no, "writing files")
    for sheet in array:
        i = Img.fromarray(sheet)
        i = i.convert("RGB")
        midfix = str(index)
        i.save("./data/tiles/"
               + str(batch_no) + "."
               + prefix * (5 - len(midfix))
               + midfix
               + suffix)
        index += 1
    print("Proc", batch_no, "done!")
    queue.put(batch_no)


def main():
    import multiprocessing as mp
    import time

    # Pulling images from HDD and slicing them up
    pool = mp.Pool(2)
    results = pool.map(img_to_slices, (fullpath, midpath))
    while len(results) != 2:
        time.sleep(0.2)
    array = np.concatenate(results, axis=0)

    # Some manual garbage collection
    del pool, results

    # This section deals with some more massive paralellism
    # The slices need to be written back to HDD, so the
    # NumPy container array is chopped up and distributed
    # to the cores with a unique identifier and each one is
    #  written to a file
    jobs = mp.cpu_count() + 2
    batches = np.array_split(array, jobs)
    results = []
    queue = mp.Queue()

    # Here we create a process for each core + 2 extra.
    # The function to be executed is passed to them as the
    # target argument along with the function's arguments
    # as a tuple.
    procs = [mp.Process(target=write_to_files, args=(batches[i], batches[i].shape[0]*i, i, queue))
             for i in range(jobs)]

    for proc in procs:
        proc.start()
    while len(results) != jobs:
        results.append(queue.get())  # The result is pulled out here
        time.sleep(0.2)
    for proc in procs:
        proc.join()


if __name__ == '__main__':
    main()
