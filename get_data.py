# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:34:47 2020

@author: Administrator
"""
import pymysql
import pandas as pd
from sshtunnel import SSHTunnelForwarder
import argparse
from datetime import datetime, timedelta
import CONSTANT
import sys

def get_data_by_sql_ssh(sql):
    '''
    线下测试用
    '''

    server = SSHTunnelForwarder(
             ('120.25.218.180', 25900),    #B机器的配置
             ssh_pkey="ssh_key",
             ssh_username="zengbin",
             remote_bind_address=('rr-wz978d24jtins2yxk.mysql.rds.aliyuncs.com', 3306))
    server.start()
    conn = pymysql.connect(host='127.0.0.1', #此处必须是是127.0.0.1
                           port=server.local_bind_port,
                           user=CONSTANT.XX_USER,
                           passwd=CONSTANT.XX_MIMA,
                           db='ty1155')
    cursor = conn.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    columns = cursor.description
    columns = [o[0] for o in columns]
    data = pd.DataFrame(data)
    data.columns = columns
    conn.close()
    server.stop()
    print('FINISH!')
    return data

def get_data_by_sql_pro(sql):
    '''
    生产环境使用
    '''
    conn = pymysql.connect(host=CONSTANT.SC_HOST,
                           port=CONSTANT.SC_PORT,
                           user=CONSTANT.SC_USER,
                           passwd=CONSTANT.SC_MIMA,
                           db=CONSTANT.SC_DB)
    cursor = conn.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    columns = cursor.description
    columns = [o[0] for o in columns]
    data = pd.DataFrame(data)
    data.columns = columns
    conn.close()
    print('FINISH!')
    return data
# First call get_sql_by_time to generate sql expression the call get_data_by_sql_pro to use this sql to pull data from the online env

# 在思路上是完全没有batching的，有可能会影响效果
def get_sql_by_time(start_y_m_d, end_y_m_d):
#     sql = f'''
#     select
#         w.id,
#         w.deliverer_id,
#         w.pay_time,
#         w.end_time,
#         w.package_geo,
#         w.sign_geo,
#         w.arrive_geo,
#         w.recv_time,
#         w.package_time,
#         ws.title,
#         w.start_time,
# --         w.ware_id,
#         w.recv_address_ext,
#         w.ware_address_ext,
#         w.sign_time,
#         w.reject_sign_time,
#         w.batch_id,
#         w.distance,
#         (LENGTH(b.waybill_ids) - LENGTH( REPLACE (b.waybill_ids, ',', ''))) + 1 as batch_num,
#         w.order_type,
#         w.vehicle_mode as vehicle_type
#     from
#         cor_waybill w
#         left join cor_warehouse_self ws on ws.id = w.self_ware_id
#         left join cor_warehouse h on h.id = w.ware_id
#         left join cor_batch b on b.id = w.batch_id
#     where w.create_time>'{start_y_m_d} 00:00:00' and w.create_time<'{end_y_m_d} 00:00:00'
#     and w.status<>'canceled'
#     and w.deliverer_id<>0
#     and (w.self_ware_id = 11 or w.self_ware_id = 18 or w.self_ware_id = 14)
#     and (w.order_type='downwind' or w.order_type='instant')
#     and (ws.title='龙华营业部' or ws.title='福田营业部' or ws.title='宝安营业部')
#     '''
    sql = f'''select distinct w.code as '运单ID',
            w.id,
            cwl.n_driver_id as '首推荐',
            a.driver_ai_id as 'AI推荐小狮哥ID',# 非自动，自动没记录
            w.deliverer_id,
            (CASE a.is_ai_driver WHEN 1 THEN '是
            ' ELSE '否' END) as '是否与推荐一致',
            a.seq as 'AI推荐订单派送序列',
            w.recv_time,
            w.pay_time,
            w.package_time as '揽件时间',
            w.start_time,
            w.sign_time,
            ws.title,
            h.title as '汽配商门店',
            w.assign_type as '派单模式',
            w.recv_address_ext,
            w.ware_address_ext,
            o.realname as '派单员',
            w.o_deliverer_id as '最后一次改派前小狮哥ID',
            w.distance,
            w.pre_end_time,
            w.assign_type as '派单方式',
            w.reject_sign_time as '拒签时间',
            w.end_time,
            (CASE WHEN (a.is_show =1 and a.opt_id=a.ai_opt_id) THEN '是' ELSE '否' END) as '是否曝光',
            w.batch_id,
            b.distance as '批次距离',
            (LENGTH(b.waybill_ids) - LENGTH( REPLACE (b.waybill_ids, ',', ''))) + 1 as '批次集单数',
            w.order_type,
            w.self_ware_id,
            w.vehicle_mode as vehicle_type
        from
            cor_waybill w
            left join ai_driver_propose a on a.waybill_id = w.id
            left join sys_operator o on a.opt_id = o.id
            left join cor_warehouse_self ws on ws.id = w.self_ware_id
            left join cor_warehouse h on h.id = w.ware_id
            left join cor_batch b on b.id = w.batch_id
            left join cor_waybill_log cwl on cwl.waybill_id = w.id
        where w.create_time>'{start_y_m_d} 00:00:00' and w.create_time<'{end_y_m_d} 00:00:00'
        and w.status<>'canceled'
        and w.deliverer_id<>0
        and (w.self_ware_id = 11 or w.self_ware_id = 18 or w.self_ware_id = 14)
        and (w.order_type='downwind' or w.order_type='instant')
        and (ws.title='龙华营业部' or ws.title='福田营业部' or ws.title='宝安营业部')
        group by w.code
        '''
    return sql


def pre_n_nay(y_m_d, n):
    y = (datetime.strptime(y_m_d, '%Y-%m-%d') - timedelta(days=n)).strftime('%Y-%m-%d')
    return y
# p for parse and f for format

if __name__ == '__main__':
    now = datetime.now().strftime('%Y-%m-%d')
    parser = argparse.ArgumentParser() # optParser is also OK,just for building a script/console App
    parser.add_argument("--start_y_m_d", type=str, default='2020-11-1')
    parser.add_argument("--end_y_m_d", type=str, default='2021-3-1')
    parser.add_argument("--num_test_days", type=int, default=3)
    parser.add_argument("--dev_mode",type=str,default="dev")
    args = parser.parse_args()
    # print(args)

    split_m_d = pre_n_nay(args.end_y_m_d, args.num_test_days)
    train_sql = get_sql_by_time(args.start_y_m_d, split_m_d)
    test_sql = get_sql_by_time(split_m_d, args.end_y_m_d)
    get_data_function = None
    if args.dev_mode == "dev":
        get_data_function = get_data_by_sql_ssh
    elif args.dev_mode == "online":
        get_data_function = get_data_by_sql_pro
    else:
        print("No mode matched")
        sys.exit(-99)
    try:
        # train_df = get_data_by_sql_pro(train_sql)
        train_df = get_data_function(train_sql)
        train_df.to_csv('data/train_raw.csv', index=False)
    except:
        print('get train data failed')
    try:
        # test_df = get_data_by_sql_pro(test_sql)
        test_df = get_data_function(test_sql)
        test_df.to_csv('data/test_raw.csv', index=False)
    except Exception as e:
        print('get test data failed')
        print(e)
    sys.exit(-2)
