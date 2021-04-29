import tensorflow as tf
import numpy as np
import pandas as pd


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float64_list=[tf.train.FloatList(value=[value])])



#*********************************
# Simple data analyst
# df2rec = pd.read_csv(input_path)
# print(df2rec.describe())
# t_df = df2rec[:30]
# print(t_df.columns)
# to_batch = pd.DataFrame(to_batch_[117])
# print(t_df.groupby(['deliverer_id']).mean())
# to_batch = t_df.sort_values(['batch_id','end_time'])#.set_index('deliverer_id')
# to_batch = dict(list(to_batch.groupby(['batch_id'],as_index=True)))
# print(to_batch)
# print(pd.DataFrame(to_batch[6572]).set_index('batch_id').drop(['end_time','start_time'],axis=1))


#*********************************
# From CSV to tfRecord
def csv_tfrecord(input_path,train_path):
    writer = tf.io.TFRecordWriter(train_path)
    df2rec = pd.read_excel(input_path) # For quick Test
    df2rec.drop(['Unnamed: 0'],axis=1,inplace=True)
    df2rec = df2rec.dropna()
    df2rec = (df2rec - df2rec.min(axis=0)) / (df2rec.max(axis=0) - df2rec.min(axis=0) + 1e-10)
    pre_data = dict(list(df2rec.groupby(['batch_id'])))
    # for k, v in pre_data.items():
    #     print(k, len(v))

    # print(df2rec)
    # df2rec['distance'] = (df2rec['distance'] - df2rec['distance'].min()) / (df2rec['distance'].max() - df2rec['distance'].min()+1e-9)
    # df2rec['dis'] = (df2rec['dis'] - df2rec['dis'].min()) / (df2rec['dis'].max() - df2rec['dis'].min()+1e-9)
    # df2rec['批次距离'] = (df2rec['批次距离'] - df2rec['批次距离'].min()) / (df2rec['批次距离'].max() - df2rec['批次距离'].min()+1e-9)
    # df2rec['汽配商门店'] = (df2rec['汽配商门店'] - df2rec['汽配商门店'].min()) / (df2rec['汽配商门店'].max() - df2rec['汽配商门店'].min()+1e-9)
    # df2rec['title'] = (df2rec['title'] - df2rec['title'].min()) / (df2rec['title'].max() - df2rec['title'].min()+1e-9)
    # df2rec['order_type'] = (df2rec['order_type'] - df2rec['order_type'].min()) / (df2rec['order_type'].max() - df2rec['order_type'].min()+1e-9)
    # df2rec['vehicle_type'] = (df2rec['vehicle_type'] - df2rec['vehicle_type'].min()) / (df2rec['vehicle_type'].max() - df2rec['vehicle_type'].min()+1e-9)
    # df2rec['deliverer_id'] = (df2rec['deliverer_id'] - df2rec['deliverer_id'].min()) / (df2rec['deliverer_id'].max() - df2rec['deliverer_id'].min()+1e-9)
    # df2rec['driver_type'] = (df2rec['driver_type'] - df2rec['driver_type'].min()) / (df2rec['driver_type'].max() - df2rec['driver_type'].min()+1e-9)
    #
    # df2rec['label'] = (df2rec['label'] - df2rec['label'].min()) / (df2rec['label'].max() - df2rec['label'].min())
    # print(len(df2rec.columns))

    # _batch_d = df2rec.sort_values(['batch_id'])

    for g_index, v in pre_data.items():

        t_df = pd.DataFrame(v)#.drop(['end_time'],axis=1)
        # print('check',g_index, t_df)
        # print(t_df.columns)
        # generate src and trg
        src_out = []
        trg_out = []
        for j in range(len(t_df)):
            tmp_src= np.array(t_df.iloc[j,1:-2])
            # print(tmp_src.shape)
            # print(tmp_src)
            src_out.append(tmp_src)
            tmp_trg = t_df.iloc[j,-2]
            # print(tmp_trg)
            trg_out.append(tmp_trg)
        _l = len(trg_out)
        # if _l < 5:
        #     try:
        #         for i in range(5-_l):
        #             src_out.append(src_out[-1])
        #             trg_out.append(trg_out[-1])
        #     except:
        #         pass
        # else:
        #     src_out=src_out[:5]
        #     trg_out = trg_out[:5]
        # _l = 5
        _src = np.concatenate(src_out,axis=0)#.reshape(-1,8)#.tostring()
        # print(_src)
        _trg = np.array(trg_out)#.reshape(-1,1)
        # Feature Extraction
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'src':_bytes_feature(_src),
            'src':tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[_src.astype(np.float64).tostring()])),
            # 'trg':_bytes_feature(_trg),
            'trg':tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[_trg.astype(np.float64).tostring()])),

            'length':_int64_feature(_l)
        }))
        writer.write(example.SerializeToString())
    writer.close()
    return


def read_decode(train_files,num_threads=2,num_epochs=1,batch_size=10,min_after_dequeue=10):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(
        train_files,
        num_epochs=num_epochs
    )
    _,serialized_example = reader.read(filename_queue)
    features_dict = tf.io.parse_single_example(
        serialized_example,
        features={
            'src':tf.io.FixedLenFeature([],tf.string),
            'trg':tf.io.FixedLenFeature([],tf.string),
            'length':tf.io.FixedLenFeature([],tf.int64),
        }
    )
    labels = tf.decode_raw(features_dict['trg'],tf.float32)
    # print(labels)
    features = tf.decode_raw(features_dict['src'],tf.float32)#[value for value in features_dict.values()]
    # features = tf.reshape(features,[-1,21])
    # 注意tf.float32会导致极端严重的维度问题,最后直接出梯度爆炸和NaN
    # must have same dType
    # feature,label = tf.train.shuffle_batch(
    #     [features,labels],
    #     batch_size=batch_size,
    #     num_threads=num_threads,
    #     capacity=min_after_dequeue+3*batch_size,
    #     min_after_dequeue=min_after_dequeue
    # )
    # res = tf.split(features,1,axis=0)
    return features,labels


def makeDataSet(file_path,batch_size):
    dataset = tf.data.TFRecordDataset(file_path)
    # tf.io.parse_single_example need data.map()
    def parse_example(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'src': tf.io.FixedLenFeature([], tf.string),
                'trg': tf.io.FixedLenFeature([], tf.string),
                'length': tf.io.FixedLenFeature([], tf.int64),
            }
        )
        l = tf.cast(features['length'],tf.int64)
        src = tf.decode_raw(features['src'],tf.float64)
        # print(src.shape)
        trg = tf.decode_raw(features['trg'],tf.float64)
        # src = tf.reshape(src,[l,8])
        # trg = tf.reshape(trg,[l])
        return (src,trg,l)
    dataset = dataset.map(parse_example)

    padded_shapes = (
        (tf.TensorShape([None])),
         tf.TensorShape([None]),
         tf.TensorShape([])
    )
    dataset.repeat(2) # Use in Training
    # dataset = dataset.shuffle(10000)
    dataset = dataset.padded_batch(batch_size,padded_shapes,drop_remainder=True)
    return dataset
# 根据batch_id整批次给测试数据，节点geo信息要预先给定

def makenpDataset(input_path):
    df = pd.read_excel(input_path)

    # print(df.min())
    # print(df.max())
    # end_time = df['end_time']
    # df = df.drop(['end_time'],axis=1)
    # 归一化
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0',axis=1,inplace=True)

    df = (df - df.min()) / (df.max() - df.min()+1e-10)
    # df['distance'] = (df['distance'] - df['distance'].min()) / (
    #             df['distance'].max() - df['distance'].min()+1e-9)
    # df['dis'] = (df['dis'] - df['dis'].min()) / (df['dis'].max() - df['dis'].min()+1e-9)
    # df['批次距离'] = (df['批次距离'] - df['批次距离'].min()) / (df['批次距离'].max() - df['批次距离'].min()+1e-9)
    # df['汽配商门店'] = (df['汽配商门店'] - df['汽配商门店'].min()) / (df['汽配商门店'].max() - df['汽配商门店'].min()+1e-9)
    # df['title'] = (df['title'] - df['title'].min()) / (df['title'].max() - df['title'].min()+1e-9)
    # df['order_type'] = (df['order_type'] - df['order_type'].min()) / (
    #             df['order_type'].max() - df['order_type'].min()+1e-9)
    # df['vehicle_type'] = (df['vehicle_type'] - df['vehicle_type'].min()) / (
    #             df['vehicle_type'].max() - df['vehicle_type'].min()+1e-9)
    # df['deliverer_id'] = (df['deliverer_id'] - df['deliverer_id'].min()) / (
    #             df['deliverer_id'].max() - df['deliverer_id'].min()+1e-9)
    # df['driver_type'] = (df['driver_type'] - df['driver_type'].min()) / (
    #             df['driver_type'].max() - df['driver_type'].min()+1e-9)
    #
    # df['label'] = (df['label'] - df['label'].min()) / (df['label'].max() - df['label'].min())

    pre_data = dict(list(df.groupby(['batch_id'])))
    dataset = []
    for g_index in pre_data.keys():
        t_df = pd.DataFrame(pre_data[g_index])
        src_out = []
        trg_out = []
        for j in range(len(t_df)):
            tmp_src = np.array(t_df.iloc[j, 1:-2])
            # print(tmp_src.shape)
            # print(tmp_src)
            src_out.append(tmp_src)
            tmp_trg = t_df.iloc[j, -2]
            # print(tmp_trg)
            trg_out.append(tmp_trg)
        _src = np.concatenate(src_out, axis=0)
        _trg = np.array(trg_out)
        b_batch = (_src,_trg,len(trg_out))
        dataset.append(b_batch)
    return dataset



if __name__ == '__main__':
    train_files = r'./result/TEST/try_multiple.tfrecords'
    csv_path = r'./data/'
    csv_name = r'concat_with_multiple_sampling.xlsx'
    input_path = csv_path + csv_name

    # df_raw = pd.read_csv(input_path)
    # df_raw.drop(['Unnamed: 0','pay_time','揽件时间'],axis=1,inplace=True)
    # # df_raw.set_index('batch_id')
    # print(df_raw.columns)
    #
    # df_geo = pd.read_csv('./data/2020年12月31日配送数据.csv')
    #
    # df_geo.drop(['Unnamed: 0'],axis=1,inplace=True)
    # df_geo.columns = ['batch_id','start_time','arrival_time','destination_a','destination_a_poi','destination_b','destination_b_poi',
    #                   'dis','ware_base','driver_type','route_details']
    # print(df_geo.columns)
    # df_geo.set_index('batch_id')
    #
    # df_merge = df_raw.merge(df_geo,on='batch_id',how='inner')
    # # df_merge = df_raw.merge(df_geo,on='batch_id',how='right')
    # df_merge.reset_index(inplace=True)
    # # print(df_merge.columns)
    # _filter = df_merge.groupby('batch_id').size()
    # index = _filter[_filter[:] > 1].index.to_list()
    # print(len(index))
    # df_data = df_merge[df_merge['batch_id'].isin(index)]
    #
    # df_data = df_data.set_index('batch_id')
    # drop_columns = ['id','运单ID','首推荐','AI推荐小狮哥ID','是否与推荐一致','AI推荐订单派送序列']
    # df_data = df_data.drop(drop_columns,axis=1)
    # df_data = df_data.sort_values(['batch_id','end_time'])
    #
    # df_data.to_excel('./data/merge_lookup.xlsx')


    # _dataset = makenpDataset(input_path)


    csv_tfrecord(input_path,train_files)
    # bd = makeDataSet(train_files)

    # batched_dataset = makeDataSet(train_files)
    # for src,trg,l in batched_dataset:
    #     print(l)

    # init_array = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True,clear_after_read=False)
    # init_array = init_array.write(0,tf.get_variable('test',shape=[1,3,2],dtype=tf.float32,initializer=tf.random_uniform_initializer))
    # print(init_array.element_shape)
    # with tf.Session() as sess:
    #     tf.global_variables_initializer()
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    #     for i in range(20):
    #         f,l = read_decode([train_files])
    #     coord.request_stop()
    #     coord.join(threads)
