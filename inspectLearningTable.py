from tkinter import *

import numpy as np

ltroot = "/data/Prog/data/learning_tables/"
ltname = "onezero.pkl.gz"


class App(Tk):
    def __init__(self):
        Tk.__init__(self)

        self.lt = pull_lt(ltroot+ltname)
        self.progress = 0

        self.title("Learning table inspector")
        self.datalab = Label(self)
        self.datalab.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.imaglab = Label(self, )
        self.imaglab.grid(row=1, column=0, columnspan=2, sticky="ew")
        Button(self, text="Previous", command=self.prev).grid(row=2, column=0, sticky="ew")
        Button(self, text="Next", command=self.next).grid(row=2, column=1, sticky="ew")

        self.load_pic()

    def next(self):
        self.progress += 1
        if self.progress > self.lt[0].shape[0]:
            self.progress = self.lt[0].shape[0]
        self.load_pic()

    def prev(self):
        self.progress -= 1
        if self.progress < 0:
            self.progress = 0
        self.load_pic()

    def load_pic(self):
        from PIL import Image, ImageTk
        img_data = np.array([self.lt[0][self.progress].reshape(60, 60) for _ in range(3)]).astype(int)
        pic = Image.fromarray(img_data, mode="RGB")
        label = self.lt[1][self.progress]
        pic = pic.resize((pic.size[0] * 5, pic.size[1] * 5))
        # pic.show()
        self.imaglab.configure(image=ImageTk.PhotoImage(pic))
        self.datalab.configure(text="Progress: {}; Divs: {}".format(self.progress, label))


def pull_lt(path):
    import pickle
    import gzip

    ltfl = gzip.open(path)
    lt = pickle.load(ltfl)
    ltfl.close()
    return lt


if __name__ == '__main__':
    app = App()
    app.mainloop()
