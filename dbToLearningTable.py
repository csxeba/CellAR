import os
import sys
import sqlite3 as sql
import pickle
import gzip

import numpy as np
from PIL import Image

dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"
dbspath = "D:/Data/dbs/" if sys.platform == "win32" else "/data/Prog/Diploma/ClassByCsa/"

sliceroot = dataroot + "raw/"
outroot = dataroot + "lts/"


def fetchbig(dbs, processing):
    recs = []
    for db in dbs:
        conn = sql.connect(dbspath + db)
        c = conn.cursor()
        c.execute("SELECT filename, divs FROM lessons WHERE divs > -1;")
        recs.extend(c.fetchall())
        conn.close()

    print("Converting slices to learning table...")
    div = np.zeros((len(recs), 60, 60, 1), dtype=np.float32)
    answ = np.zeros((len(recs),), dtype=np.int32)

    for i, rec in enumerate(recs):
        path = sliceroot + processing + rec[0]
        img = Image.open(path)
        div[i] = np.array(img)[..., [0]]
        answ[i] = int(rec[1])

    print("Created BIG learning table in memory")
    return div.reshape((div.shape[0], 1, 60, 60)), answ


def fetchsmall(dbs, processing):
    recs = []
    for db in dbs:
        conn = sql.connect(dbspath + db)
        c = conn.cursor()
        c.execute("SELECT filename, divs FROM lessons WHERE divs > 0;")
        positive = c.fetchall()
        c.execute("SELECT filename, divs FROM lessons WHERE divs = 0;")
        zero = c.fetchall()[:len(positive)]
        recs.extend(positive + zero)
        conn.close()

    print("Converting slices to learning table...")
    div = np.zeros((len(recs), 60, 60, 1), dtype=np.float32)
    answ = np.zeros((len(recs),), dtype=np.int32)

    for i, rec in enumerate(recs):
        path = sliceroot + processing + rec[0]
        img = Image.open(path)
        div[i] = np.array(img)[..., [0]]
        answ[i] = int(rec[1])

    print("Created SMALL learning table in memory")
    return div.reshape((div.shape[0], 1, 60, 60)), answ


def fetchonezero(dbs, processing):
    recs = []
    for db in dbs:
        conn = sql.connect(dbspath + db)
        c = conn.cursor()
        c.execute("SELECT filename, divs FROM lessons WHERE divs = 1;")
        ones = c.fetchall()
        c.execute("SELECT filename, divs FROM lessons WHERE divs = 0;")
        zeros = c.fetchall()[:len(ones)]
        conn.close()
        recs.extend(ones + zeros)

    print("Converting slices to learning table...")
    div = np.zeros((len(recs), 60, 60, 1), dtype=np.float32)
    answ = np.zeros((len(recs),), dtype=np.int32)

    for i, rec in enumerate(recs):
        path = sliceroot + processing + rec[0]
        img = Image.open(path)
        div[i] = np.array(img)[..., [0]]
        answ[i] = int(rec[1])

    print("Created 01 learning table in memory")
    return div.reshape((div.shape[0], 1, 60, 60)), answ


def fetchxonezero(dbs, processing):
    ones = []
    zeros = []
    for db in dbs:
        conn = sql.connect(dbspath + db)
        c = conn.cursor()
        c.execute("SELECT filename, divs FROM lessons WHERE divs = 1;")
        ones.extend(c.fetchall())
        c.execute("SELECT filename, divs FROM lessons WHERE divs = 0;")
        zeros.extend(c.fetchall())
        conn.close()

    print("Converting slices to learning table...")
    div1 = np.zeros((len(ones), 60, 60, 1), dtype=np.float32)
    div1m = np.zeros((len(ones), 60, 60, 1), dtype=np.float32)
    div0 = np.zeros((len(ones) * 2, 60, 60, 1), dtype=np.float32)
    answ1 = np.ones((len(ones) * 2,), dtype=np.int32)
    answ0 = np.zeros_like(answ1, dtype=np.int32)

    for i, rec in enumerate(ones):
        path = sliceroot + processing + rec[0]
        img = Image.open(path)
        div1[i] = np.array(img)[..., [0]]
        div1m[i] = np.fliplr(div1[i])
    i = 0
    while i < len(ones) * 2:
        path = sliceroot + processing + zeros[i][0]
        img = Image.open(path)
        div0[i] = np.array(img)[..., [0]]
        i += 1

    print("Created X01 learning table in memory")
    questions = np.concatenate((div1, div1m, div0))
    answers = np.concatenate((answ1, answ0))

    return questions.reshape((questions.shape[0], 1, 60, 60)), answers


def generate_all():
    prc = ["tiles", "ctr", "convd", "bgs"]
    data = ["big", "small", "onezero", "xonezero"]

    for processing in prc:
        for dataset in data:
            method = {"b": fetchbig, "s": fetchsmall, "o": fetchonezero, "x": fetchxonezero}[dataset[0]]
            lt = method(os.listdir(dbspath), processing + "/")

            f = gzip.open(outroot + dataset + "_" + processing + ".pkl.gz", "wb")
            print("Dumping", dataset + "-" + processing)
            pickle.dump(lt, f)
            f.close()
    print("Done!")

if __name__ == '__main__':
    fetchsmall(os.listdir(dbspath), "raw")
