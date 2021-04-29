from get_data import *
import pandas as pd
import numpy as np


# sql =  "select * from bc_driver_geo where locate_time > '2021-2-1' and locate_time < '2021-2-2'"
sql = get_sql_by_time('2021-2-1','2021-2-2')
df = get_data_by_sql_ssh(sql)
print(df.columns)
df.to_excel('./data/train_raw.xlsx')


station_position = {
    '11': (114.045637, 22.641598),
    '14': (114.103832, 22.561315),
    '18': (113.88749, 22.575465),
    '27': (114.075657, 22.621867),
    '23': (113.923672, 22.687786),
    '10482': (114.068214, 22.630206),
    '21': (113.906205, 22.515095),
    '22': (113.944063, 22.55986),
    '26': (114.105479, 22.559681),
    '15': (113.324981, 23.150597),
    '16': (113.269114, 22.988374),
    '17': (113.244417, 23.138102),
    '19': (114.234343, 22.70001),
    '25': (113.84875, 22.736349),
    '32': (114.027992, 22.558885),
}


# df = pd.read_csv('./data/geo_data.csv')
# print(df.columns)
# df_gp = df.sort_values('created_time')#.set_index('batch_id')
# print(df_gp[:10].drop('Unnamed: 0',axis=1))


# *************************************************
# def read_only_db_con():
#   return create_engine("mysql+pymysql://isuser:JHbs_VsdhgFS970sVjhs@rr-wz978d24jtins2yxk.mysql.rds.aliyuncs.com:3306/ty1155?charset=utf8")
#
#
# def get_raw_trace(date, engine):
#     sql =  f"select * from bc_driver_geo where date(locate_time)='{date}'"
#     return pd.read_sql_query(sql, engine)

# engine = read_only_db_con()
# df = get_raw_trace(date="2021-03-29", engine=engine)
