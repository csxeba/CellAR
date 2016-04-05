import sqlite3 as sql

pixpath = "D:/Data/tiles/"
dbpath = "./db/"

conn = sql.connect(dbpath + "data.db")
cur = conn.cursor()

command = "SELECT * FROM lessons WHERE id < 10"

cur.execute(command)
res = cur.fetchall()
for ln in res:
    print(ln)
