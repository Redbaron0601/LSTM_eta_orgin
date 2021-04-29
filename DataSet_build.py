import pandas as pd
from preprocess import preprocess_time
import numpy as np
from datetime import datetime as dt
from math import radians, sin, cos, asin, sqrt
from preprocess import filter_extrem
import json


#
# g_df_list = []
# iter_batch = df.iterrows()
# raw_df = [d for d in iter_batch]


def get_label_by_day(y,m,d,df):
    _label = []
    _dis = []
    ix = 0
    df_len = len(df)

    df = preprocess_time(df)
    _index = df.index

    batch_num = df.groupby('batch_id').size()
    batch_num.name = 'batch_num'
    batch_num = pd.DataFrame(batch_num).reset_index()
    df.drop(['派单员', '最后一次改派前小狮哥ID', '派单方式', 'index', '是否曝光', '批次集单数'], axis=1, inplace=True)
    df = df.merge(batch_num,on='batch_id',how='left')


    # df['recv_address_ext'] = df['recv_address_ext'].apply(lambda x: json.load(x)['location'])
    df['ware_address_ext'] = df['ware_address_ext'].apply(lambda x: json.loads(x)['location'])
    df.drop('start_time_y',axis=1,inplace=True)

    while ix < df_len:
        try:
            _jmp = int(np.sqrt(df['batch_num'][ix]))
        except:
            ix += int(df['batch_num'][ix])
            continue

        _label.extend([(df['end_time'][ix]-df['start_time_x'][ix])/np.timedelta64(60,'s')]*_jmp)
        b_pos= df['destination_b_poi'][ix].strip('(').strip(')').replace(' ','').replace('(','').replace(')','').split(',')
        a_pos = df['destination_a_poi'][ix].strip('(').strip(')').replace(' ','').replace('(','').replace(')','').split(',')
        _dis.extend([geodistance(a_pos[0],a_pos[1],b_pos[0],b_pos[1])]*_jmp)

        for i in range(2,_jmp+1):
            for i_x in range(ix+(i-1)*_jmp,min(ix+i*_jmp,df_len)):
                # print(df['sign_time'][i_x-_jmp])
                if str(df['sign_time'][i_x-_jmp]) != 'NaT':
                    _label.append((df['end_time'][i_x] - df['sign_time'][i_x - _jmp]) / np.timedelta64(60, 's'))
                else:
                    # print(str(df['sign_time'][i_x-_jmp]))
                    _label.append((df['end_time'][i_x] - df['拒签时间'][i_x - _jmp]) / np.timedelta64(60, 's'))

                b_pos = df['destination_b_poi'][i_x].strip('(').strip(')').replace(' ', '').split(',')
                a_pos = df['destination_a_poi'][i_x].strip('(').strip(')').replace(' ', '').split(',')
                _dis.append(geodistance(a_pos[0],a_pos[1],b_pos[0],b_pos[1]))
        ix += _jmp*_jmp
    # print(_label)
    print(len(_label))
    print(df_len)
    _min = min(df_len,len(_label))
    label = pd.DataFrame(_label[:_min],index=_index,columns=['label'])
    _dis = pd.DataFrame(_dis[:_min],index=_index,columns=['_dis'])
    #
    df__ = df[:_min]
    df__ = df__.join(_dis)
    __df = df__.join(label)
    __df.drop('distance',axis=1,inplace=True)

    __df.to_excel(f'./data/label_{m}_{d}.xlsx')

    return __df
    # _label = []
    # # df.reset_index(inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # group = df.groupby(by="batch_id")
    #
    # full_data = []
    # invalid_data_rows = []
    #
    # for id, (gid, g) in enumerate(group):
    #
    #     _jmp = g['批次集单数'].unique()[0]
    #     cnt = 0
    #     m = g['sign_time'].values.tolist()
    #     if None in m:
    #         invalid_data_rows.extend(g["index"].index.values.tolist())
    #         continue
    #     a = (g['end_time'] - g['start_time_x']) / np.timedelta64(60, 's')
    #     _label.extend(a.values[:_jmp])
    #     cnt += 1
    #
    #     for i in range(len(g)):
    #         index = g["index"].index.values
    #         if i // _jmp >= 1:
    #             size = i // _jmp
    #             _label.append((g['end_time'][index[i]] - g['sign_time'][index[size]]) / np.timedelta64(60, 's'))
    #
    #     # g.reset_index(inplace=True)
    #     full_data.append(g)
    #
    # df = pd.concat(full_data)
    #
    # label = pd.DataFrame({"label": _label})
    #
    # df = pd.concat([df, label], axis=1)
    # df = df.drop(labels=invalid_data_rows, axis=0)
    # df = df[df["label"] != np.nan]
    #
    # df.reset_index(inplace=True, drop=True)
    # # df = df.join(label)
    # # df = df[df['label']>np.timedelta64(0)]
    # #
    # df.to_excel('./data/label_one_day.xlsx')




def str2dt(date_str):
    try:
        # dt.fromisoformat(date_str)
        if "-" in date_str:
            if len(date_str) > 19:
                return dt.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
            else:
                return dt.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        else:
            if len(date_str) > 19:
                return dt.strptime(date_str, "%Y/%m/%d %H:%M:%S.%f")
            else:
                return dt.strptime(date_str, "%Y/%m/%d %H:%M:%S")
    except:
        print("error datestr:",date_str)
        # 可以考虑拯救一下数据，毕竟合理的采样值有点稀少
        return


def geodistance(lng1, lat1, lng2, lat2):
    lng1 = lng1.replace("'","")
    lat1 = lat1.replace("'","")
    lng2 = lng2.replace("'","")
    lat2 = lat2.replace("'","")
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance , 3)
    return distance


# def Simple_Sample(df_label):
#     ix = 0
#     cur_sample = []
#
#     df_label['new_label'] = [0 for i in range(df_label.shape[0])]
#     df_label['pre_speed'] = [0 for i in range(df_label.shape[0])]
#     df_label['pre_time'] = [0 for i in range(df_label.shape[0])]
#     df_rows = df_label.iterrows()
#     rows = [row for row in df_rows]
#
#     while ix < len(df_label):
#         try:
#             batch_num = int(np.sqrt(df_label['batch_num'][ix]))
#         except:
#             ix += int(df_label['batch_num'][ix])
#             continue
#         # print(batch_num)
#         s_c = np.random.randint(0, batch_num-1)
#         # print(s_c)
#
#         # Choose which process to do Sampling
#         # Lately will deal with the speed ISSUE!
#         i = ix+s_c*batch_num+s_c
#         while i < min(ix + batch_num*batch_num,len(df_label)):
#             # Choose TimeStamp for specific node
#             pools = df_label['route_details'][i].split("(Timestamp('")[1:]
#             try:
#                 pos = list(map(lambda x: x.split("'),")[1].strip().rstrip('),').replace(' ',''), pools))
#             except:
#                 i += batch_num + 1
#                 continue
#             pools = list(map(lambda x: x.split("'), ")[0].replace("'(",""), pools))
#             pools = list(map(lambda x: str2dt(x),pools))
#             max_len = max(2,len(pools))
#             p_node = np.random.randint(0, max_len // 2)
#             stamp_time = pools[p_node]
#             geo_pos = pos[p_node].strip('(').strip(')').replace('(','').replace(')','').strip(']').split(',')
#             dest_pos = df_label['destination_b_poi'][i].strip('(').strip(')').replace(' ','').strip(']').split(',')
#             src_pos = df_label['destination_a_poi'][i].strip('(').strip(')').replace(' ','').strip(']').split(',')
#
#
#
#             if i<len(df_label):
#                 df_label["new_label"][i] = s_c#(rows[i][1]['end_time'] - stamp_time)/np.timedelta64(60,'s')
#                 df_label["label"][i]=(rows[i][1]['end_time'] - stamp_time)/np.timedelta64(60,'s')
#
#                 if df_label["label"][i]==np.NaN:
#                     i += batch_num + 1
#                     continue
#                 df_label['_dis'][i] = geodistance(geo_pos[1], geo_pos[0], dest_pos[0], dest_pos[1])
#                 # df_label['pre_speed'] = geodistance(src_pos[0],src_pos[1],geo_pos[1],geo_pos[0])
#                 # if i >= batch_num:
#                 #     df_label['pre_time'] = (stamp_time-rows[i-batch_num][1]['sign_time'])/np.timedelta64(60,'s')
#                 # else:
#                 #     df_label['pre_time'] = (stamp_time - rows[i][1]['recv_time'])/np.timedelta64(60,'s')
#                 df_label['pre_time'] = (stamp_time - rows[i][1]['recv_time']) / np.timedelta64(60, 's')
#                 try:
#                     df_label['pre_dis'][i] = geodistance(src_pos[0],src_pos[1],geo_pos[1],geo_pos[0])#/(df_label['pre_time'][i])
#                 except:
#                     i += (batch_num + 1)
#                     continue
#                 cur_sample.append(df_label.iloc[i][:])
#             elif i>=len(df_label):
#                 break
#             i += (batch_num+1)
#         ix = ix + batch_num*batch_num
#     cur_df = pd.DataFrame(cur_sample)
#     return cur_df

def Simple_Sample(df_label):
    cur_sample = []
    # df_label['new_label'] = [0 for i in range(df_label.shape[0])]
    # df_label['pre_speed'] = [0 for i in range(df_label.shape[0])]
    df_label['pre_dis'] = [0 for i in range(df_label.shape[0])]
    for i in range(len(df_label)):
        pools = df_label['route_details'][i].split("(Timestamp('")[1:]
        try:
            pos = list(map(lambda x: x.split("'),")[1].strip().rstrip('),').replace(' ',''), pools))
        except:
            i += 1
            continue
        pools = list(map(lambda x: x.split("'), ")[0].replace("'(", ""), pools))
        pools = list(map(lambda x: str2dt(x), pools))
        max_len = max(2, len(pools))
        p_node = np.random.randint(max_len//4, max_len // 2)
        stamp_time = pools[p_node]

        df_label["label"][i] = (rows[i][1]['end_time'] - stamp_time) / np.timedelta64(60, 's')
        geo_pos = pos[p_node].strip('(').strip(')').replace('(', '').replace(')', '').strip(']').split(',')
        dest_pos = df_label['destination_b_poi'][i].strip('(').strip(')').replace(' ','').strip(']').split(',')
        src_pos = df_label['destination_a_poi'][i].strip('(').strip(')').replace(' ','').strip(']').split(',')

        df_label["label"][i] = (rows[i][1]['end_time'] - stamp_time) / np.timedelta64(60, 's')
        df_label['_dis'][i] = geodistance(geo_pos[1], geo_pos[0], dest_pos[0], dest_pos[1])

        try:
            df_label['pre_dis'][i] = geodistance(src_pos[0],src_pos[1],geo_pos[1],geo_pos[0])#/(df_label['pre_time'][i])
        except:
            i += 1
            continue
        cur_sample.append(df_label.iloc[i][:])
    return pd.DataFrame(cur_sample)



def target_Encoding(df):
    df_label = df[df['label'].notnull()]
    encoding_cols = ['title', 'deliverer_id', 'order_type', 'driver_type','vehicle_type']
    valid_cols = ['batch_id', 'deliverer_id', 'title',  '_dis', 'pre_dis','dis',
                  '批次距离', 'batch_num', 'order_type', 'driver_type','vehicle_type', 'label']
    df_label_ = df_label[valid_cols]
    for col in encoding_cols:
        mean_label = df_label_[[col, 'label']].groupby(by=[col])["label"].mean()
        df_label_[col] = df_label_[col].apply(lambda x: mean_label[x])

    return df_label_


if __name__ == '__main__':

    # csv_path = r'./data/'
    # csv_name = r'label_one_day.xlsx'
    # input_path = csv_path + csv_name

    # df_raw = pd.read_excel('./data/train_raw.xlsx')
    # df_raw.drop(['Unnamed: 0', 'pay_time', '揽件时间','汽配商门店'], axis=1, inplace=True)
    # # df_raw.set_index('batch_id')
    # print(df_raw.columns)
    #
    # df_geo = pd.read_excel('./data/oneday_path.xlsx')    # pre label
    #
    # df_geo.drop(['Unnamed: 0'], axis=1, inplace=True)
    # df_geo.columns = ['batch_id', 'start_time', 'arrival_time', 'destination_a', 'destination_a_poi', 'destination_b',
    #                   'destination_b_poi',
    #                   'dis', 'ware_base', 'driver_type', 'route_details']
    # print(df_geo.columns)
    # # df_geo.set_index('batch_id')
    #
    # df_merge = df_raw.merge(df_geo, on='batch_id', how='inner')
    # # df_merge = df_raw.merge(df_geo, on='batch_id', how='right')
    # df_merge.reset_index(inplace=True)
    # # print(df_merge.columns)
    # _filter = df_merge.groupby('batch_id').size()
    # index = _filter[_filter[:] > 1].index.to_list()
    # print(len(index))
    # df_data = df_merge[df_merge['batch_id'].isin(index)]
    #
    # df_data = df_data.set_index('batch_id')
    # drop_columns = ['id', '运单ID', '首推荐', 'AI推荐小狮哥ID', '是否与推荐一致', 'AI推荐订单派送序列']
    # df_data = df_data.drop(drop_columns, axis=1)
    # df_data = df_data.sort_values(['batch_id', 'end_time'])
    #
    # df_data.to_excel('./data/merge_lookup.xlsx')

    # df = pd.read_excel('./data/merge_lookup.xlsx')
    df = pd.read_excel('./data/clean_label.xlsx')

    df_rows = df.iterrows()
    rows = [row for row in df_rows]

    y = 2021
    m = 1
    d = 1

    df_label= get_label_by_day(y,m,d,df)

    # 循环随机采样增强数据*********************************
    # cur = None
    #
    # for k in range(100):
    #     cur = Simple_Sample(df)
    #
    #     cur = target_Encoding(cur)
    #
    #     cur = filter_extrem(cur, 'zt_sw')
    #
    #     cur.to_excel(f'./data/eval/sample-{k}.xlsx')
    # **************************************************

    # df_label = pd.read_excel('./data/label_one_day.xlsx')

        # df = pd.read_excel('./data/label1.xlsx')
        # g_df_list.append(Simple_Sample(df))
    # cur = Simple_Sample(df_label)
    #
    # cur = target_Encoding(cur)
    # cur = filter_extrem(cur,'zt_sw')
    #
    # cur.to_excel(f'./data/labels-{y}-{m}-{d}.xlsx')

    # new_group = cur.groupby(by=['batch_id', 'label'])
    # unique_data = []
    # for id, g in new_group:
    #     # index = g["index"].index.values
    #     unique_data.append(g[0:1])
    #
    # df_label_ = pd.concat(unique_data)
    # df_label_.reset_index(inplace=True, drop=True)
    # df_label_.to_excel('final_feats_label.xlsx')
    #
    # df_label_.to_excel(f'./data/labels-{y}-{m}-{d}.xlsx')


# Start Sampling
# def Simple_Sample(df_label):
#     ix = 0
#     cur_sample = []
#     while ix < len(df_label):
#         batch_num = df_label['批次集单数'][ix]
#         print(batch_num)
#         s_c = np.random.randint(0,batch_num)
#         # Choose which process to do sampling
#         # Maybe deal with the speed ISSUE
#         for i in range(ix+s_c,ix+batch_num):
#             # Choose TimeStamp for specific node
#             pools = df_label['route_details'][ix+s_c].split(', (')
#             max_len = len(pools)
#             p_node = np.random.randint(0,max_len//2)
#             cur_time = pools[p_node]
#             print(cur_time)
#             _time = cur_time.split('(')[1].split(')')[0]
#             print(_time)
#             stamp_time = pd.to_datetime(_time)
#             if i==(ix+s_c):
#                 df_label['label'][ix+s_c] = (df_label['end_time'][ix+s_c]-stamp_time)
#         ix = ix+batch_num
#     return df_label
