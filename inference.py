from lstm_model import LSTM_Model,CHECKPOINT_PATH
from transform2Tfrecord import makenpDataset
import tensorflow as tf
import numpy as np



train_files = './data/eval/2021-2-2-labels.xlsx'
if __name__ == '__main__':
    with tf.variable_scope('LSTM_model',reuse=None):
        model = LSTM_Model()

    dataset = makenpDataset(train_files)

    # Test
    # _init = tf.global_variables_initializer()

    # Simulate One time Inference
    loss_first_node = []
    for _s,_t,_l in dataset[:-1]:
    # _s,_t,_l = dataset[0]

        eta_op = model.inference(_s,_l)

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT_PATH)
    # sess.run(init)
    # Inference
        infer_eta = sess.run([eta_op])
        # print(infer_eta)
        # print(_t)
        print((infer_eta[0][0]-_t[0])/infer_eta[0][0])
        if (infer_eta[0][0]-_t[0])/infer_eta[0][0] == np.inf:
            continue
        loss_first_node.append(abs((infer_eta[0][0]-_t[0])/infer_eta[0][0]))
    # print(loss_first_node)
    print(sum(loss_first_node)/len(loss_first_node))
