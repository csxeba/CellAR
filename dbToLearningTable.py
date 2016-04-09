import os
import random
import sqlite3 as sql

import numpy as np
from PIL import Image


dbspath = "/data/Prog/Diploma/GorApps/"
slicepath = "/data/Prog/data/raw/convd/"


def fetchrecs(db):
    conn = sql.connect(dbspath + db)
    c = conn.cursor()
    c.execute("SELECT filename, divs FROM lessons WHERE divs > 0;")
    recs = c.fetchall()
    c.execute("SELECT filename, divs FROM lessons WHERE divs == 0;")
    zeros = c.fetchall()
    conn.close()
    # random.shuffle(zeros)
    # recs.extend(zeros[:len(recs)])
    recs.extend(zeros)
    random.shuffle(recs)
    print("Fetched {} records!".format(len(recs)))

    return recs


def slices_to_lessons(recbatch):
    print("Converting slices to learning table...")
    questions = np.zeros((len(recbatch), 60, 60, 1), dtype=np.float32)
    answers = np.zeros((len(recbatch,)), dtype=int)
    for i, rec in enumerate(recbatch):
        path = slicepath + rec[0]
        img = Image.open(path)
        questions[i] = np.array(img)[..., [0]]
        answers[i] = int(rec[1])
    print("Created learning table in memory")
    return questions.reshape((len(recbatch), 1, 60, 60)), answers


def lessons_to_learning_table(less):
    questions = np.concatenate([l[0] for l in less])
    answers = np.concatenate([l[1] for l in less])
    return questions, answers


def dump_learning_table(lt):
    import pickle
    import gzip
    f = gzip.open("learning.table.pkl.gz", "wb")
    print("Dumping...")
    pickle.dump(lt, f)
    f.close()
    print("Done!")


if __name__ == '__main__':
    records = []    
    flz = os.listdir(dbspath)
    for recset in map(fetchrecs, flz):
        records += recset
    lessons = slices_to_lessons(records)
    dump_learning_table(lessons)
    print("Learning table saved as learning_table.pkl.gz")
