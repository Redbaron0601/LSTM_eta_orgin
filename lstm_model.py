import tensorflow as tf
from transform2Tfrecord import makeDataSet
import numpy as np

# input path
train_files = r'./result/TEST/try.tfrecords'
# Model output path
CHECKPOINT_PATH = './model/lstm_ckpt/rnn_ckpt-1600'
OUT_PUT_CKPT = './model/lstm_ckpt/rnn_ckpt'
log_dir = './result/log'
# ********************
# SETTING PARAMS
BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
HIDDEN_SIZE = 32
hidden_layer = 16
NUM_LAYERS = 5
FEATURE_LENGTH = 11
MAX_GRAD_NORM = 5
NUM_EPOCH = 10
# ********************
class LSTM_Model(object):
    def __init__(self):

        self.cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE,state_is_tuple=True))
             for _ in range(NUM_LAYERS)]
        )

        self.eta_estimate_1 = tf.get_variable('rnn_out_1',[HIDDEN_SIZE,hidden_layer],dtype=tf.float32)
        self.eta_estimate_2 = tf.get_variable('rnn_out_2',[hidden_layer,1],dtype=tf.float32)
        self.eta_estimate = tf.reshape(self.eta_estimate_1,[-1,1])
        self.eta_bias_1 = tf.get_variable('rnn_bias_1',[hidden_layer],dtype=tf.float32)
        self.eta_bias_2 = tf.get_variable('rnn_bias_2',[1],dtype=tf.float32)


    def forward(self,input,target,size,type='train'):
        batch_size = tf.shape(input)[0]
        # input = tf.reshape(input, [5, 5, -1])
        d2 = tf.shape(input)
        input = tf.cast(input, tf.float32)
        size = tf.cast(size, tf.float32)
        target = tf.cast(target, tf.float32)

        input = tf.reshape(input,[batch_size,-1,FEATURE_LENGTH])
        size = tf.reshape(size,[batch_size])

        with tf.variable_scope('eta_estimator'):
            outputs, state = tf.nn.dynamic_rnn(
                self.cell,input,size,dtype=tf.float32
            )

            output = tf.reshape(outputs,[-1,HIDDEN_SIZE])
            eta = tf.nn.relu(tf.matmul(output,self.eta_estimate_1) + self.eta_bias_1)
            eta = tf.nn.tanh(tf.matmul(eta,self.eta_estimate_2) + self.eta_bias_2)
            eta = tf.reshape(eta,[-1])

        label_weights = tf.sequence_mask(size, maxlen=tf.shape(target)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])

        if type == 'train':
            loss = cal_huber(eta,tf.reshape(target,[-1]))

            cost = tf.reduce_sum(loss * label_weights)
            cost_per_node = cost / tf.to_float(batch_size)
            tf.summary.scalar('eta-estimator/losses', cost_per_node)

            trainable_vars = tf.trainable_variables()
            grads = tf.gradients(cost_per_node, trainable_vars)

            grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
            optimizer = tf.train.AdamOptimizer(learning_rate=5e-4)
            train_op = optimizer.apply_gradients(
                zip(grads, trainable_vars)
            )
            merged = tf.summary.merge_all()

            return eta, merged,cost_per_node,train_op
        else:
            test_loss_1 = smape(tf.reshape(target, [-1]), eta)
            test_loss_2 = delay(tf.reshape(target, [-1]), eta)

            cost_1 = tf.reduce_sum(test_loss_1 * label_weights)#/tf.reduce_sum(label_weights)
            cost_2 = tf.reduce_sum(test_loss_2 * label_weights)#/tf.reduce_sum(label_weights)
            cost_mape = cost_1 / tf.to_float(batch_size)
            cost_delay = cost_2 / tf.to_float(batch_size)

            return cost_mape,cost_delay


    def inference(self,input,size):
        # Dynamic Call is OK
        # batch_size = tf.shape(input)[0]
        # input = tf.cast(input, tf.float32)
        # size = tf.cast(size, tf.float32)
        # label = tf.cast(label, tf.float32)
        #
        # input = tf.reshape(input, [TEST_BATCH_SIZE, -1, 8])
        # size = tf.reshape(size, [batch_size])
        #
        #
        # with tf.variable_scope('eta_estimator'):
        #     infer_outs, _ = tf.nn.dynamic_rnn(
        #         self.cell,input,size,dtype=tf.float32
        #     )
        #     infer_out = tf.reshape(infer_outs,[-1,HIDDEN_SIZE])
        #     eta = tf.nn.relu(tf.matmul(infer_out, self.eta_estimate_1) + self.eta_bias_1)
        #     eta = tf.reshape(eta, [-1])
        #
        #     test_loss_1 = smape(tf.reshape(label,[-1]),eta)
        #     test_loss_2 = delay(tf.reshape(label,[-1]),eta)
        #
        #     label_weights = tf.sequence_mask(size, maxlen=tf.shape(label)[1], dtype=tf.float32)
        #     label_weights = tf.reshape(label_weights, [-1])
        #     cost_1 = tf.reduce_sum(test_loss_1 * label_weights) / tf.reduce_sum(label_weights)
        #     cost_2 = tf.reduce_sum(test_loss_2 * label_weights) / tf.reduce_sum(label_weights)
        #     cost_mape = cost_1 / tf.to_float(batch_size)
        #     cost_delay = cost_2 / tf.to_float(batch_size)
        #
        # return cost_mape,cost_delay
        input = tf.cast(input, tf.float32)
        size = tf.cast(size, tf.float32)


        input = tf.reshape(input,[1,-1,FEATURE_LENGTH])
        size = tf.reshape(size,[-1])

        with tf.variable_scope('eta_estimator'):
            outputs, state = tf.nn.dynamic_rnn(
                self.cell,input,size,dtype=tf.float32
            )

            output = tf.reshape(outputs,[-1,HIDDEN_SIZE])
            eta = tf.nn.relu(tf.matmul(output,self.eta_estimate_1) + self.eta_bias_1)
            eta = tf.nn.tanh(tf.matmul(eta,self.eta_estimate_2) + self.eta_bias_2)
            eta = tf.reshape(eta,[-1])
        return eta


def cal_huber(eta,label,delta=1.0):
    res = tf.abs(label-eta)
    sym = tf.subtract(label,eta)
    condition_1 = tf.less(res,delta)
    condition_2 = tf.less(sym,0.0)
    gamma = 8.0
    loss = tf.where(condition_1,tf.where(condition_2,gamma*0.5*tf.square(res),0.5*tf.square(res)),tf.where(condition_2,gamma*(delta*res-0.5*tf.square(delta)),delta*res-0.5*tf.square(delta)))
    return loss


def smape(label,eta):
    diff_abs = tf.abs(label-eta)
    diff_base = tf.abs(eta)
    return diff_abs/(diff_base)


def delay(label,eta):
    _delay = tf.abs(eta-label)
    # zeros = tf.zeros([tf.shape(label)[0]])
    # cond = tf.less(_delay,0.0)
    # return tf.where(cond,_delay,zeros)
    return _delay


def run_epoch(sess,ops,saver,step,writer,type='train'):
    mape_sum = 0.0
    delay_sum = 0.0
    while True:
        try:
            if type=='train':
                eta, summery, cost, _ = sess.run(ops)
                print(eta)
                if step % 10 == 0:
                    print("After %d steps, per node cost is %.3f" % (step, cost))
                if step % 200 == 0:
                    saver.save(sess, OUT_PUT_CKPT, global_step=step)
                writer.add_summary(summery,step)
            else:
                _mape,_delay = sess.run(ops)
                mape_sum += _mape
                delay_sum += _delay

            step += 1
        except tf.errors.OutOfRangeError:
            if type=='test':
                print("Evaluating MAE Loss: %.5f Per Day" % (mape_sum/(step+1)))
                print("Evaluating Delay: %.4f Per Day" % (delay_sum/(step+1)))
                return mape_sum/(step+1), delay_sum/(step+1)
            break
    return step


# def run_epoch(sess,merge_op,cost_op,train_op,saver,step,type='train'):
#     while True:
#         try:
#             summery, cost, _, = sess.run([merge_op,cost_op,train_op])
#             if step % 10 == 0:
#                 print("After %d steps, per node cost is %.3f" % (step, cost))
#             if step % 200 == 0:
#                 saver.save(sess, OUT_PUT_CKPT, global_step=step)
#             writer.add_summary(summery,step)
#             step += 1
#         except tf.errors.OutOfRangeError:
#             break
#     return step



if __name__ == '__main__':
    # init = tf.random_uniform_initializer(-0.1,0.1)
    with tf.variable_scope('LSTM_model',reuse=None):
        model = LSTM_Model()

    batched_dataset = makeDataSet(train_files,50) # Use test data

    # Train
    _it = batched_dataset.make_initializable_iterator()
    (s, t, l) = _it.get_next()
    merge_op, cost_op, train_op = model.forward(s, t, l,'train')


    saver = tf.train.Saver()
    step = 0

    with tf.Session() as sess:
        # saver.restore(sess,CHECKPOINT_PATH)
        writer = tf.summary.FileWriter(log_dir,sess.graph)
        tf.global_variables_initializer().run()

        ops = [merge_op,cost_op,train_op]
        for i in range(NUM_EPOCH):
            print("In Iteration: %d" % (i+1))
            sess.run(_it.initializer)
            step = run_epoch(sess,ops,saver,step,writer,'train')
        writer.close()


    # Test
    # _init = tf.global_variables_initializer()

    # itest = batched_dataset.make_one_shot_iterator()
    # _s, _t, _l = itest.get_next()
    # _cost1,_cost2 = model.inference(_s, _t, _l)
    #
    # sess = tf.Session()
    # saver = tf.train.Saver()
    # saver.restore(sess, CHECKPOINT_PATH)
    # # sess.run(init)
    # # Inference
    # _cost_1,_cost_2 = sess.run([_cost1,_cost2])
    # print(_cost_1,_cost_2)
