import sqlite3
import os
import pandas as pd
from PySources.mergeTable import create_table, get_tb_names
import numpy as np


def sample_data_with_rate(data: pd.DataFrame, rate: float):
    n = int(np.ceil(len(data) * rate))
    return data.sample(n=n)


def filter_unique_formula_value(db_path: str, critical_col: str):
    db_temp = db_path[:-3] + "_temp.db"
    try: os.remove(db_temp)
    except: pass

    conn = sqlite3.connect(db_temp)
    curs = conn.cursor()
    conn_ori = sqlite3.connect(db_path)
    curs_ori = conn_ori.cursor()

    #
    curs_ori.execute(get_tb_names())
    list_table = [tb_ for tb_ in curs_ori.fetchall() if tb_[0].startswith("T")]
    print(list_table)
    # raise
    list_table = list(map(lambda t: int(t[0][1:]), list_table))
    for i in list_table:
        curs.execute(create_table(i, critical_col))
    conn.commit()

    for i in list_table:
        curs_ori.execute(f"select count(*) from T{i};")
        num_rows = curs_ori.fetchone()[0]
        print(i, num_rows)
        rate = 100000.0 / num_rows

        curs_ori.execute(f"SELECT * FROM T{i};")
        list_df = []
        while True:
            data = curs_ori.fetchmany(1000000)
            if not data:
                break
            data = pd.DataFrame(data)
            data["temp"] = data[2].round(3)
            temp = data.groupby("temp", group_keys=False).apply(lambda x: sample_data_with_rate(x, rate))
            list_df.append(temp)

        data = pd.concat(list_df, ignore_index=True)
        rate = 100000.0 / len(data)
        list_to_ins = data.groupby("temp", group_keys=False).apply(lambda x: sample_data_with_rate(x, rate))
        list_to_ins = list_to_ins.drop(columns=["temp"]).values.tolist()
        print(i, len(list_to_ins))
        curs.executemany(f"INSERT INTO T{i} VALUES(?, ?, ?)", list_to_ins)
        conn.commit()

    conn.close()
    conn_ori.close()
