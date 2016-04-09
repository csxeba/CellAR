from tkinter import *
import tkinter.messagebox as tkmb
import sqlite3 as sql
from PIL import Image, ImageTk
import os

pixpath = "/data/Prog/data/raw/tiles/"
dbpath = "/data/Prog/Diploma/GorApps/"
dbname = "sum.db"


class DB:
    def __init__(self, master):
        self.conn = sql.connect(dbpath + dbname)
        self.cur = self.conn.cursor()
        self.master = master
        self.create = "CREATE TABLE lessons " + \
                      "(id int primary key, filename text, divs int);"

    def initialize(self):

        welcome_drink = "Üdvözöllek a programban!\n" + \
                        "Úgy tűnik, most nyitottad meg először. Most el fog " + \
                        "indulni egy inicializációs folyamat, ami eltarthat " + \
                        "egy kis ideig. Dobok egy ablakot, ha kész van."

        tkmb.showinfo("Üdv a programban!", welcome_drink)

        self._create_empty_db()
        no_records = self._fill_db()

        tkmb.showinfo("Figyelem!",
                      "Az inicializáció befejeződött. " +
                      "Előre is köszönöm a segítséget, kezdheted a munkát!")
        self.master.files = no_records

    def modrec(self, ID, divs):
        params = (divs, ID)
        update = "UPDATE lessons SET divs=? WHERE id==?;"
        self.cur.execute(update, params)
        self.conn.commit()
        print("Updated lesson no {}, divs={}".format(ID, divs))

    def reset(self):
        print("Reseted")
        self.cur.execute("RENAME TABLE lessons TO lessonsbak")
        self.cur.execute("DROP TABLE lessons")

    def _create_empty_db(self):
        try:
            self.cur.execute(self.create)
        except sql.OperationalError:
            # self.reset()
            # self.cur.execute(self.create)
            pass

    def _fill_db(self):
        flz = sorted(os.listdir(pixpath))
        print("Adding {} records to database".format(len(flz)))
        for ID, fl in enumerate(flz):
            insert = "INSERT INTO lessons VALUES (?, ?, ?);"
            self.cur.execute(insert, (ID, fl, "-1"))
        self.conn.commit()
        return len(flz)


class App(Tk):
    def __init__(self):
        Tk.__init__(self)

        # Define object attributes
        self.buttons = [None for _ in range(10)]  # holds the Button instances indexed by their value
        self.selected = -1  # refers to the button currently pressed
        self.pic = None  # hold the picture currently inspected
        self.progress, self.files = pull_meta()  # n_o picture curr opened and n_o all the pictures
        self.db = DB(self)  # refers to the database wrapper
        self.proglabel = None  # holds the label that displays the progress
        self.container = None  # this is a label instance that displays the pic curr inspected

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

        Button(buttonframe, text="Mi a feladat?", font=f,
               command=self._info
               ).grid(row=6, column=1, columnspan=cols+1, sticky="ew")
        Button(buttonframe, text="Következő", font=f, width=w,
               command=self.next
               ).grid(row=6, column=cols+2, sticky="ew")

        self.bind("<Key>", self._keypress)
        self.bind("<Return>", self.next)
        self.bind("<Left>", self.back)
        self.bind("<Right>", self.next)
        # GUI built

        # Initialize database if neccesary
        if (not self.files) or \
           (self.files != len(os.listdir(pixpath))) or \
           (self.files < self.progress):

            self.db.initialize()
            self.progress = 0
        # Database initialized

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
        self.db.modrec(ID=self.progress, divs=self.selected)
        self.progress += 1
        if self.progress == self.files:
            self.progress = self.files - 1
            return

        self._label_update()
        self._getpic()
        self._update_meta()
        self._update_buttons()

    def back(self, event=None):
        del event
        self.progress -= 1
        if self.progress < 0:
            self.progress = 0
            return

        self._label_update()
        self._getpic()
        self._update_meta()
        self._update_buttons()

    def _label_update(self):
        progchain = "Haladás: " + str(self.progress) + " / " + str(self.files)
        self.proglabel.configure(text=progchain)

    def _keypress(self, key):
        """Converts keypress event to "button event" """
        x = key.char
        if x in [str(i) for i in range(10)]:
            self._selection(int(x))

    def _getpic(self):
        select = "SELECT filename FROM lessons WHERE id==?"
        self.db.cur.execute(select, (str(self.progress),))
        flname = self.db.cur.fetchone()[0]
        
        print("Opening", pixpath + flname)
        pic = Image.open(pixpath + flname)
        pic = pic.resize((pic.size[0]*5, pic.size[1]*5))
        self.pic = ImageTk.PhotoImage(pic)
        self.container.configure(image=self.pic)

    def _update_buttons(self):
        self.db.cur.execute("SELECT divs FROM lessons WHERE id==?", (str(self.progress),))
        self._selection(self.db.cur.fetchone()[0])

    def _selection(self, x):
        x = int(x)
        self.selected = x if 0 <= x < 10 else -1
        for key, button in enumerate(self.buttons):
            button.configure(relief=RAISED)
        self.buttons[x].configure(relief=SUNKEN)

    def _good_exit(self):
        self._update_meta()
        self.destroy()

    def _update_meta(self):
        chain = ""
        for m, val in zip(("Progress", "Files"), (self.progress, self.files)):
            chain = chain + m + ": " + str(val) + "\n"

        fl = open(dbpath + ".meta.dat", mode="w")
        fl.write(chain)
        fl.close()

    @staticmethod
    def _info():
        tkmb.showinfo("Információ", helptext)


def freshen():
    print("Adatbázis törlése...")
    flz = os.listdir(dbpath)
    for fl in flz:
        os.remove(dbpath+fl)


def pull_meta():
    if ".meta.dat" not in os.listdir(dbpath):
        fl = open(dbpath + ".meta.dat", mode="w")
        fl.write("Progress: 0\nFiles: 0")
        fl.close()

    fl = open(dbpath + ".meta.dat", mode="r")
    chain = fl.read()
    fl.close()

    d = {key: int(val) for key, val in [ln.split(": ") for ln in chain.split("\n")[:2]]}

    return d["Progress"], d["Files"]


def sanity_check():
    dbdir = os.listdir(dbpath)
    if len(dbdir) == 0:  # no database file and no metadata file is present
        return "fresh"
    elif len(dbdir) == 1:
        prg, flz = pull_meta()
        if (flz == 0) and (prg == 0):
            return "fresh"
        else:
            return "broken"
    elif len(dbdir) == 2:
        prg, flz = pull_meta()
        if (flz == 0) and (prg == 0):
            return "fresh"
        elif flz > prg >= 0:
            if flz != len(os.listdir(pixpath)):
                return "broken"
            else:
                return "progressed"
        else:
            return "broken"
    else:
        return "broken"


def brokenstate():
    v = tkmb.askyesno("Ajjaj",
                      "Sajnálattal közölnöm kell, hogy ez a programverzió lehet, hogy" +
                      "működésképtelen...\n" +
                      "Viszont ha most először indítottad el a programot, akkor " +
                      "lehet, hogy helyre lehet rázni.\n" +
                      "Először nyitottad meg a programot?")
    if v:
        freshen()
    else:
        tkmb.showerror("Ajjaj", "Kérned kellene tőlem egy másik programot...\n" +
                       "+3630-789-2809 vagy csxeba@gmail.com az elérhetőségem.")
        import sys
        sys.exit()


helptext = """A szakdolgozatomat abból szeretném írni, hogy lehetséges-e az itt látható \
formátumú képeken egy mesterséges intelligencia ágens felhasználásával \
megszámoltatni az osztódásban lévő sejteket.

Egy mesterséges intelligencia ágens működőképessé tételéhez meg kell neki \
tanítani előzetesen, hogy mi az, amit fel kell ismernie. A tanulótábla egy \
manuálisan előre "megfejtett" adatsor kell, hogy legyen. Jelenleg ennek a \
tanulótáblának a létrehozásában kérem a segítségedet. Minél több lecke áll az \
ágens rendelkezésére, annál pontosabban tud majd működni.

A feladat az lenne, hogy megszámold, hány osztódásban lévő sejtet látsz a képen.
Ha valamiben nem vagy biztos, fontos, hogy inkább hagyd ki, majd utólag ellenőrzöm,
hogy mi maradt ki és azokat osztályozom én vagy kihagyom a végső táblából.

A program könyvtárában láthatsz egy "Leiras" pdf-et, ott bejelöltem néhány \
osztódást, illetve pozitív és negatív példákat hozok.

A programot irányíthatod a látható gombok lenyomásával, a billentyűzettel vagy \
a numerikus billentyűzettel is. Választást jóváhagyni az Enterrel és az \
előre gombbal is lehet, visszalépni pedig a visszagombbal."""

if __name__ == '__main__':
    # state = sanity_check()
    #
    # if state is "broken":
    #     brokenstate()
    # elif state is "fresh":
    #     freshen()

    app = App()
    app.mainloop()
