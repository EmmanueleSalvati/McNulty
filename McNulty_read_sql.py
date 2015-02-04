"""test to connect to TestStress"""
from McNulty_config import connect
import pandas.io.sql as psql
import pickle as pkl


def import_df(sql_table):
    """Import dataframe from MySql, returns a pandas dataframe"""

    db = connect()
    sql = "SELECT * FROM %s" % sql_table
    df = psql.read_sql(sql, db)
    db.close()

    return df


if __name__ == '__main__':

    df = import_df('all_hospitals')
    with open('hospitals.pkl', 'w') as pklfile:
        pkl.dump(df, pklfile)
