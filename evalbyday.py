import os
import pandas as pd
from lstm_model import *
import tensorflow as tf
from transform2Tfrecord import makeDataSet,csv_tfrecord

#******************************
TEST_FILE_PATH = r'./data/eval/'

# file_pattern = r'^2021-*'
#******************************
File_queue = []

with tf.variable_scope('LSTM_model', reuse=None):
    model = LSTM_Model()

for file in os.listdir(TEST_FILE_PATH):
    # print(type(file))
    if '-labels.xlsx' in file:
        File_queue.append(file)

__mae = []
__delay = []

for _file in File_queue:
    file_path = TEST_FILE_PATH + _file

    csv_tfrecord(file_path,'./data/eval/cur.tfrecords')

    test_per_day = makeDataSet('./data/eval/cur.tfrecords',10)

    _it = test_per_day.make_one_shot_iterator()
    (s, t, l) = _it.get_next()
    ops = model.forward(s, t, l, type='eval')

    saver = tf.train.Saver()
    step = 0


    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_PATH)
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        mae,_delay = run_epoch(sess,ops,saver,step,writer,'test')
        __mae.append(mae)
        __delay.append(_delay)

        writer.close()

mae_df = pd.DataFrame(__mae,columns=['MAE'])
delay_df = pd.DataFrame(__delay,columns=['TIME DELAY'])

res_df = mae_df.join(delay_df)
res_df.to_csv('./data/eval/eval_with11.csv')
