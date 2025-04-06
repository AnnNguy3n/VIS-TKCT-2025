import sqlite3
import os
import numpy as np
import gc


def decode(lst):
    return "_".join(map(str, lst))


def query_table(table_num, num_opr, critical_col):
    cols = ["id"] + list(map(lambda x: f"E{x}", range(num_opr))) + [critical_col]
    return f"SELECT {','.join(cols)} FROM T{table_num}_{num_opr};"


def create_table(table_num, critical_col):
    return f"CREATE TABLE IF NOT EXISTS T{table_num} (id INTEGER, Formula TEXT, {critical_col} REAL);"


def get_tb_names():
    return 'SELECT name FROM sqlite_master WHERE type = "table";'


NUM_FML_PROCESS = [0, 70, 6090, 114380, 5984720, 28126028, 3945059720, 3314820620]


def merge_table(db_path: str, critical_col: str):
    assert db_path.endswith("f.db")
    new_path = db_path[:-4] + "f_new.db"
    try: os.remove(new_path)
    except: pass

    connect_origin = sqlite3.connect(db_path)
    connect_new = sqlite3.connect(new_path)
    cursor_origin = connect_origin.cursor()
    cursor_new = connect_new.cursor()

    #
    cursor_origin.execute(get_tb_names())
    list_table = cursor_origin.fetchall()
    list_table = [t[0] for t in list_table if t[0].startswith("T")]
    temp = list(map(lambda x: int(x.split("_")[0][1:]), list_table))
    list_tbname = np.unique(temp)

    for table_num in list_tbname:
        cursor_new.execute(create_table(table_num, critical_col))
        connect_new.commit()
        temp = [t for t in list_table if t.startswith(f"T{table_num}_")]
        for t in temp:
            num_opr = int(t.split("_")[1])
            cursor_origin.execute(query_table(table_num, num_opr, critical_col))
            n = 0
            bias = sum(NUM_FML_PROCESS[:num_opr])
            while True:
                rows = cursor_origin.fetchmany(10000000)
                if not rows: break

                data_to_insert = [(row[0]+bias, decode(row[1:-1]), row[-1]) for row in rows]
                cursor_new.executemany(f"INSERT INTO T{table_num} VALUES (?, ?, ?);", data_to_insert)
                connect_new.commit()
                n += len(rows)
                print(table_num, num_opr, n)
                del rows, data_to_insert
                gc.collect()

    #
    connect_origin.close()
    connect_new.close()