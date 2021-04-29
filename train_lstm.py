from lstm_model import LSTM_Model,CHECKPOINT_PATH,log_dir,NUM_EPOCH,run_epoch
from transform2Tfrecord import makeDataSet
import tensorflow as tf


train_files = './result/TEST/try_multiple.tfrecords'
if __name__ == '__main__':
    # init = tf.random_uniform_initializer(-2,2)
    with tf.variable_scope('LSTM_model', reuse=None):
        model = LSTM_Model()

    batched_dataset = makeDataSet(train_files, 30)  # Use test data

    # Train
    _it = batched_dataset.make_initializable_iterator()
    (s, t, l) = _it.get_next()
    ops = model.forward(s, t, l, type='train')

    saver = tf.train.Saver()
    step = 0

    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_PATH)
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        # tf.global_variables_initializer().run()

        for i in range(NUM_EPOCH):
            print("In Iteration: %d" % (i + 1))
            sess.run(_it.initializer)
            step = run_epoch(sess, ops, saver, step, writer,type='train')
        writer.close()
