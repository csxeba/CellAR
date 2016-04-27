import os
import sqlite3 as sql

import numpy as np
from PIL import Image

from csxnet.datamodel import shuffle

dbspath = "/data/Prog/Diploma/ClassByCsa/"
sliceroot = "/data/Prog/data/raw/"
processing = "tiles"
slicepath = sliceroot + processing + "/"


def fetchrecs(db):
    conn = sql.connect(dbspath + db)
    c = conn.cursor()
    c.execute("SELECT filename, divs FROM lessons WHERE divs = 1;")
    ones = c.fetchall()
    c.execute("SELECT filename, divs FROM lessons WHERE divs = 0;")
    zeros = c.fetchall()
    conn.close()
    print("Fetched {} records!".format(len(ones)+len(zeros)))

    return ones, zeros


def slices_to_lessons(ones, zeros):
    print("Converting slices to learning table...")
    div1 = np.zeros((len(ones), 60, 60, 1), dtype=np.float32)
    div1m = np.zeros((len(ones), 60, 60, 1), dtype=np.float32)
    div0 = np.zeros((len(ones) * 2, 60, 60, 1), dtype=np.float32)
    answ1 = np.ones((len(ones) * 2,), dtype=np.int32)
    answ0 = np.zeros_like(answ1, dtype=np.int32)

    for i, rec in enumerate(ones):
        path = slicepath + rec[0]
        img = Image.open(path)
        div1[i] = np.array(img)[..., [0]]
        div1m[i] = np.fliplr(div1[i])
    i = 0
    while i < len(ones)*2:
        path = slicepath + zeros[i][0]
        img = Image.open(path)
        div0[i] = np.array(img)[..., [0]]
        i += 1

    print("Created learning table in memory")
    questions = np.concatenate((div1, div1m, div0))
    answers = np.concatenate((answ1, answ0))
    return questions.reshape((questions.shape[0], 1, 60, 60)), answers


def dump_learning_table(lt):
    import pickle
    import gzip
    f = gzip.open("learning.table_" + processing + ".pkl.gz", "wb")
    print("Dumping...")
    pickle.dump(lt, f)
    f.close()
    print("Done!")


if __name__ == '__main__':
    ones, zeros = [], []
    for ons, zrs in map(fetchrecs, os.listdir(dbspath)):
        ones += ons
        zeros += zrs
    lessons = slices_to_lessons(ones, zeros)
    lessons = shuffle(lessons)
    dump_learning_table(lessons)
    print("Learning table saved as learning_table.pkl.gz")
