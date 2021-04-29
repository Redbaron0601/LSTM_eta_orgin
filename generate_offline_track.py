# -*- coding: utf-8 -*-
# @Date    : 2021/2/22 19:51
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import pandas as pd
# import common
from gui_util import *
import copy
from math import radians, cos, sin, asin, sqrt


day10s = {}

df_data = []


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

def get_file(file_tmp=r'./data/geo_data.csv'):
    path = file_tmp
    return path


def get_one_day_csv(file_tmp=r'./data/train_raw.csv'):
    auto = pd.read_csv(file_tmp)
    # auto = auto[]
    auto = auto[auto['批次集单数'] > 1]
    # auto = auto[auto['title'].str.contains('龙华')]  # 84
    return auto


def get_route_detail(on_way):
    route_detail = [(s[0],s[1], s[2]) for s in on_way]
    return route_detail


def get_pair_info(day_df_data, pair, time_position, driver_id):
    """

    Args:
        pair: 每一个订单的起点终点信息
        time_position: list，订单的记录时间

    Returns:

    """



    for last, this_o in pair:
        b_id = last['batch_id']
        start_time = last['end_time']
        arrival_time = this_o['end_time']
        destination_a = last['维修厂']
        destination_a_poi = last["维修厂地址"]
        destination_b = this_o['维修厂']
        destination_b_poi = this_o["维修厂地址"]
        driver_type = this_o["骑手类型"] # 骑手类型
        ware_base = last['self_ware_id']
        on_way = get_on_way_data(time_position, start_time, arrival_time, driver_id)

        # on_way = log_lon_lat_clean(on_way)
        # 计算距离
        dis = get_dis(on_way)
        if len(on_way) == 0:
            continue

        route_details = get_route_detail(on_way)

        bi_dist = 0

        # 上一次订单 到达时间 ， 这一个订单时间， 上一个维修厂， 当前维修厂 ， 实际轨迹距离， 导航距离
        row_t = (start_time, arrival_time, destination_a, destination_b, dis, bi_dist, driver_type,ware_base)
        day10s[(destination_a, destination_b)] = (
            start_time, arrival_time, destination_a, destination_b, dis, bi_dist, driver_type)
        print(row_t)
        day_df_data.append(
            (b_id,start_time, arrival_time, destination_a, destination_a_poi, destination_b, destination_b_poi,
             dis, ware_base, driver_type, route_details))

    return day_df_data


def convert_batch_info_to_seq_info(auto_t):
    """
    将每一行的订单信息，提取成sequence的数据
    Args:
        auto_t:

    Returns:

    """

    def read_json(json):
        if isinstance(json, str):
            true = True
            null = None
            false = False
            return eval(json)['name']
        return '?'

    def get_lng_lat_repair(json):
        if isinstance(json, str):
            true = True
            null = None
            false = False
            return (eval(json)['location']['lng'], eval(json)['location']['lat'])
        return '?'

    auto_t['维修厂'] = auto_t['recv_address_ext'].apply(read_json)
    auto_t['维修厂地址'] = auto_t['recv_address_ext'].apply(get_lng_lat_repair)
    auto_t = auto_t[['batch_id','id', 'deliverer_id', 'start_time', 'end_time', 'distance', '维修厂', '维修厂地址',"骑手类型",'self_ware_id']]
    # row_ds = [dict(d) for ix, d in auto_t.iterrows()]
    b_id = auto_t['batch_id'].values[0]
    # print(b_id)
    bs_time = auto_t['start_time'].values[0]

    row_ds = [d for ix, d in auto_t.iterrows()]
    # print(type(row_ds))
    pair = []
    ware_pair = {}
    for i, row_d in enumerate(row_ds):  # 组合成起点，终点格式
        # 相邻的两单放一起
        if i == 0:
            try:
                print(row_d['self_ware_id'])
                ware_pos = station_position[str(row_d['self_ware_id'])]
                ware_pair.setdefault('batch_id',b_id)
                ware_pair.setdefault('end_time',bs_time)
                ware_pair.setdefault('维修厂',str(row_d['self_ware_id']))
                ware_pair.setdefault('维修厂地址',ware_pos)
                ware_pair.setdefault('骑手类型',row_d['骑手类型'])
                ware_pair.setdefault('self_ware_id',str(row_d['self_ware_id']))
                pair.append((ware_pair,row_d))
                continue
            except KeyError as e:
                print('Wrong self_ware_id!')
            # continue

        pair.append((row_ds[i - 1], row_d))
    return pair


def get_batch_info(day_df_data, df_align, auto, ex, driver_id):
    """
    按每一单进行处理
    Args:
        df_align: 原始对齐的数据表格
        auto: 读取的自动派单站点的数据（龙华）
        ex: 当前订单

    Returns:

    """

    # 过滤 批次
    # df_align2 = df_align[df_align['batch_id'] == ex]
    df_align2 = df_align[df_align['driver_id'] == driver_id]
    if len(df_align2)==0:
        return day_df_data
    df_align_ = df_align2.set_index(keys=["driver_id"])
    driver_type_dict = df_align_.to_dict()["driver_type"]
    driver_type = driver_type_dict[driver_id]
    # 改数据格式
    time_position = [(pd.Timestamp(a), b, c) for a, b, c in
                     zip(df_align2['locate_time'], df_align2['lat'], df_align2['lng'])]

    auto_t = auto[auto['batch_id'] == ex]
    auto_t = auto_t.sort_values(['end_time'])

    auto_t["骑手类型"] = driver_type
    # auto_t: 给定单号的订单信息
    pair = convert_batch_info_to_seq_info(auto_t)

    print('pair', pair)
    day_df_data = get_pair_info(day_df_data, pair, time_position, driver_id=driver_id)
    return day_df_data



def get_one_day_summary(y, m, d,df,train_raw):
    '''
    Args:
        y: 年
        m: 月
        d: 日
    Returns:

    '''

    import os
    day = d
    # path = get_file()
    # df = pd.read_csv(path)
    # 目前从bc_driver_geo表中抽取的单日数据，量级在25w左右
    print('geo len', len(df))
    print(df.columns)
    # df.columns = ['Unnamed: 0', 'id', 'driver_id', 'geo', 'speed', 'direction', 'release', 'brand', 'model', 'c_time',
    #               'created_time']
    # geo = df["geo"].values.tolist()
    # geo_lat =geo.copy()

    # df["lat"] = [eval(item)["lat"] for item in geo_lat]
    # df["lng"] = [eval(item)["lng"] for item in geo]
    # lngs = []

    # for item in geo:
    #     # print(eval(item))
    #     try:
    #         lngs.append(eval(item)["lng"])
    #
    #     except:
    #         print(eval(item))


    # print('geo batch numb', len(df['batch_id'].unique()))
    # 按时间排序
    df = df.sort_values(['created_time'])
    dir = f'./data'
    tmp = dir + '/train_raw.csv'

    auto = train_raw#get_one_day_csv(file_tmp=tmp)

    print('len auto batch', len(auto['batch_id'].unique()))
    assert len(auto['batch_id'].unique()) > 1
    try:
        df_align = df[df['batch_id'].isin(auto['batch_id'].unique())]
    except:
        df_align = pd.DataFrame()
    if len(df_align) == 0:
        df_align = df

    day_df_data = []
    driver_ids = auto["deliverer_id"].unique()
    for id, ex in enumerate(auto['batch_id'].unique()):
        # 对每一个批次，更新全局数据表
        driver_id = auto[auto['batch_id'] == ex]["deliverer_id"].unique()[0]
        day_df_data = get_batch_info(day_df_data, df_align, auto, ex, driver_id)

    return day_df_data


if __name__ == "__main__":
    day = 1
    month = 2
    year = 2021
    raw = pd.read_csv('./data/train_raw.csv')
    _df = pd.read_csv('./data/geo_data.csv')
    df_data = get_one_day_summary(year, month, day,_df,raw)
    df = pd.DataFrame(df_data)
    df.to_excel('./data/oneday_path.xlsx')
    # print(df.columns)

    # df_g = df.groupby(0).size()
    # index = df_g[df_g[:] > 1].index.to_list()
    # df_data = df.loc[df[0].isin(index)]
    # print(df_data)
    # print(df_g[df_g[:]>1])

    # df_data.to_excel(f'./data/{year}年{month}月{day}日配送数据.xlsx')
    # df.to_excel(f'./data/{year}年{month}月{day}日配送数据.xlsx')
