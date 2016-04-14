import os
from tkinter import *
import sqlite3 as sql

from PIL import Image, ImageTk

dbpath = "/data/Prog/Diploma/ClassByCsa/"
dbname = "all.db"
pixpath = "/data/Prog/data/raw/tiles/"


class DB:
    def __init__(self):
        self.conn = sql.connect(dbpath + dbname)
        self.cur = self.conn.cursor()
        self.progress = 0

    def initialize(self):
        create = "CREATE TABLE lessons (id int, filename text, divs int);"
        insert = "INSERT INTO lessons (id, filename, divs) VALUES (?, ?, ?);"
        print("Initializing database...")
        try:
            self.cur.execute(create)
        except sql.OperationalError:
            print("DB already exists, skipping initialize()")
            return

        print("Initialized DB @ " + dbpath + dbname)
        flz = sorted(os.listdir(pixpath))
        for i, flnm in enumerate(flz):
            self.cur.execute(insert, (i, flnm, -1))

    def modrec(self, id_, divs):
        update = "UPDATE lessons SET divs=? WHERE id=?;"
        self.cur.execute(update, (str(divs), str(id_)))
        print("Rec {} set to {}".format(id_, divs))

    def select(self, id_):
        command = "SELECT * FROM lessons WHERE id=?"
        self.cur.execute(command, str(id_))
        rec = self.cur.fetchone()
        return rec


class App(Tk):
    def __init__(self):
        Tk.__init__(self)

        # Define object attributes
        self.buttons = [None for _ in range(10)]  # holds the Button instances indexed by their value
        self.pic = None  # hold the picture currently inspected
        self.proglabel = None  # holds the label that displays the progress
        self.container = None  # this is a label instance that displays the pic curr inspected
        self.progress = 0
        self.files = 0

        self.db = DB()
        self.db.initialize()

        # Building the GUI
        self.title("Diplomamunka feladat")

        head = Label(self, text="Tanulótábla-generáló", font=14)
        self.proglabel = Label(self, text=" / ", font=14)
        self.container = Label(self)
        buttonframe = Frame(self)
        controlframe = Frame(self)

        f = 16

        cols = 3
        counter = 1
        for rown in range(2, -1, -1):
            for coln in range(cols):
                coln += 1
                b = Button(buttonframe, width=3, text=counter, font=f,
                           command=lambda x=counter: self._selection(x))
                b.grid(row=rown, column=coln)
                self.buttons[counter] = b
                counter += 1

        b = Button(buttonframe, text="0", font=f,
                   command=lambda x=0: self._selection(x))
        b.grid(row=4, column=1, columnspan=cols+1, sticky="ew")
        self.buttons[0] = b
        b = Button(buttonframe, text="Semmi", font=f,
                   command=lambda x=10: self._selection(x))
        b.grid(row=5, column=1, columnspan=cols+1, sticky="ew")
        self.buttons.append(b)

        w = 10

        Button(buttonframe, text="Vissza", font=f, width=w,
               command=self.back).grid(row=6, column=0, sticky="ew")

        Button(buttonframe, text="Mi a feladat?", font=f
               ).grid(row=6, column=1, columnspan=cols+1, sticky="ew")
        Button(buttonframe, text="Következő", font=f, width=w,
               command=self.next
               ).grid(row=6, column=cols+2, sticky="ew")

        self.bind("<Key>", self._keypress)
        self.bind("<Return>", self.next)
        self.bind("<Left>", self.back)
        self.bind("<Right>", self.next)
        # GUI built

        # Update pic holder
        self._getpic()
        self._label_update()

        # Display the created widgets
        head.pack()
        self.proglabel.pack()
        self.container.pack()
        buttonframe.pack()
        controlframe.pack()

        # Bind exit method to WM close button
        self.protocol("WM_DELETE_WINDOW", self._good_exit)

        self._update_buttons()

    def next(self, event=None):
        del event
        self.db.modrec(self.progress, self.selected)
        self.progress += 1
        if self.progress >= 9006:
            self.progress = 9006


        self._label_update()
        self._getpic()
        self._update_buttons()

    def back(self, event=None):
        del event
        self.progress -= 1
        if self.progress < 0:
            self.progress = 0
            return

        self._label_update()
        self._getpic()
        self._update_buttons()

    def _label_update(self):
        progchain = "Haladás: " + str(self.progress)
        self.proglabel.configure(text=progchain)

    def _keypress(self, key):
        """Converts key(board)press event to "button event" """
        x = key.char
        if x in [str(i) for i in range(10)]:
            self._selection(int(x))

    def _getpic(self):
        flname = self.db.select(self.progress)[1]
        print("Opening", pixpath + flname)
        pic = Image.open(pixpath + flname)
        pic = pic.resize((pic.size[0]*5, pic.size[1]*5))
        self.pic = ImageTk.PhotoImage(pic)
        self.container.configure(image=self.pic)

    def _update_buttons(self):
        self._selection(self.db.select(self.progress)[2])

    def _selection(self, x):
        x = int(x)
        self.selected = x if 0 <= x < 10 else -1
        for key, button in enumerate(self.buttons):
            button.configure(relief=RAISED)
        self.buttons[x].configure(relief=SUNKEN)

    def _good_exit(self):
        self.destroy()

if __name__ == '__main__':
    app = App()
    app.mainloop()
