from get_data import get_sql_by_time,get_data_by_sql_ssh
from generate_offline_track import *
from DataSet_build import get_label_by_day,target_Encoding,Simple_Sample
from transform2Tfrecord import csv_tfrecord
from preprocess import  filter_extrem

#*****PATH AND CONSTANTS******
READ_IN = r'./data/train_raw.csv'
GEO_PATH = r'./data/geo_data.csv'
WRITE_TO = r'./data/pre_label.csv'
MON_DAY = {
    1:31,
    2:28,
    3:31,
    4:30,
    5:31,
    6:30,
    7:31,
    8:31,
    9:30,
    10:31,
    11:30,
    12:31
}
#*****************************
def main():
    train_time_base = [2]
    df_concat = pd.DataFrame()

    for mon in train_time_base:
        for day in range(1,MON_DAY[mon]):
            if day==MON_DAY[mon]:
                if mon==12:
                    sql = get_sql_by_time(f'2020-{mon}-{day}', f'2021-1-1')
                    geo_sql = "select * from bc_driver_geo where locate_time > '2020-{}-{}' and locate_time < '2021-{}-{}'".format(
                        mon, day, 1, 1)
                else:
                    sql = get_sql_by_time(f'2021-{mon}-{day}', f'2021-{mon+1}-1')
                    geo_sql = "select * from bc_driver_geo where locate_time > '2021-{}-{}' and locate_time < '2021-{}-{}'".format(
                        mon, day, mon+1, 1)
            else:
                sql = get_sql_by_time(f'2021-{mon}-{day}', f'2021-{mon}-{day+1}')
                geo_sql = "select * from bc_driver_geo where locate_time > '2021-{}-{}' and locate_time < '2021-{}-{}'".format(
                    mon,day,mon,day+1)
            try:
                train_raw = get_data_by_sql_ssh(sql)
                # train_raw.to_csv(READ_IN)
                geo_data = get_data_by_sql_ssh(geo_sql)
                print(f"Get geo 2021-{mon}-{day} is OK")
            except Exception as e:
                print(f"Get DATA 2021-{mon}-{day} FAILED!!!")
                print(e)
                # raise ValueError("Wrong Data access")
                continue
            geo_one_day = get_one_day_summary(2021, mon, day, geo_data, train_raw)
            geo_df = pd.DataFrame(geo_one_day)
            df_g = geo_df.groupby(0).size()
            index = df_g[df_g[:] > 1].index.to_list()
            df_geo = geo_df.loc[geo_df[0].isin(index)]

            # Merge data ******************************
            train_raw.drop(['pay_time', '揽件时间'], axis=1, inplace=True)
            if 'Unnamed: 0' in train_raw.columns:
                train_raw.drop(['Unnamed: 0'], axis=1, inplace=True)
            if 'Unnamed: 0' in df_geo.columns:
                df_geo.drop(['Unnamed: 0'], axis=1, inplace=True)
            df_geo.columns = ['batch_id', 'start_time', 'arrival_time', 'destination_a', 'destination_a_poi',
                              'destination_b',
                              'destination_b_poi',
                              'dis', 'ware_base', 'driver_type', 'route_details']
            # print(df_geo.columns)
            df_merge = train_raw.merge(df_geo, on='batch_id', how='inner')
            df_merge.reset_index(inplace=True)

            _filter = df_merge.groupby('batch_id').size()
            index = _filter[_filter[:] > 1].index.to_list()
            print(len(index))
            df_data = df_merge[df_merge['batch_id'].isin(index)]

            # df_data = df_data.set_index('batch_id')
            drop_columns = ['运单ID', '首推荐', 'AI推荐小狮哥ID', '是否与推荐一致', 'AI推荐订单派送序列']
            df_data = df_data.drop(drop_columns, axis=1)
            df_data = df_data.sort_values(['batch_id', 'end_time'])
            # print(df_data.columns)

            df_data.to_excel(f'./merge/merge_lookup-{mon}-{day}.xlsx')

            del(df_data)

            _df = pd.read_excel(f'./merge/merge_lookup-{mon}-{day}.xlsx')
            if 'Unnamed: 0' in _df.columns:
                _df.drop('Unnamed: 0',axis=1,inplace=True)
            # print(_df.columns)
            # _df.reset_index(drop=True,inplace=True)

            df_ = get_label_by_day(2021, mon, day, _df)
            df_.to_excel(f'./data/resample-{mon}-{day}.xlsx')
            # for _ in range(10):
            #     cur = Simple_Sample(df_)
            #
            #     cur = target_Encoding(cur)
            #
            #     cur = filter_extrem(cur,'zt_sw')

            # new_group = cur.groupby(by=['batch_id', 'label'])
            # unique_data = []
            # for id, g in new_group:
            #     # index = g["index"].index.values
            #     unique_data.append(g[0:1])
            #
            # df_label_ = pd.concat(unique_data)
            # df_label_.reset_index(inplace=True, drop=True)

            # print(cur.columns)
            # cur.to_excel(f'./data/2020-{mon}-{day}-labels.xlsx')

            # df_label_.to_excel('final_feats_label.xlsx')
            #     df_concat = pd.concat([df_concat, cur], axis=0)
            # if (day+1) % MON_DAY[mon] == 0:
            #     df_concat.to_excel('./data/concat_with_multiple_sampling.xlsx')




if __name__ == '__main__':
    main()
