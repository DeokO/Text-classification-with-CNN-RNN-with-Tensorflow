####################################################
# Text classification with RNN - model (character level)
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.07.
####################################################



#####################################
# Import modules
#####################################
import gc
gc.collect()
import tensorflow as tf
from Ch03_TextRNN.Text_RNN_config import *
import Ch01_Data_load.utils as utils



################################################################################
# Network Scratch!
################################################################################
class MODEL():

    def __init__(self, sess, JM, FLAGS):

        ####################################################
        # Jaso mapping
        ####################################################
        self.sess = sess
        self.JM = JM

        ####################################################
        # Set placeholders
        ####################################################
        self.set_placeholders()

        ####################################################
        # Network structure
        ####################################################
        self.set_network()

        ####################################################
        # Optimizer / accuracy / loss / tensorboard
        ####################################################
        self.set_ops(FLAGS.WRITER, FLAGS.WRITER_generate)


    ########################################################
    # Set placeholder
    ########################################################
    def set_placeholders(self):
        self.X_Onehot = tf.placeholder(dtype=tf.string, shape=[None, FLAGS.MAXLEN])
        self.X = tf.placeholder(tf.float32, [None, FLAGS.INPUT_WIDTH, FLAGS.INPUT_DEPTH])
        self.Y = tf.placeholder(tf.float32, [None, FLAGS.NUM_OF_CLASS])
        self.SEQ = tf.placeholder(tf.int32, [None])
        self.LEARNING_RATE = tf.placeholder(tf.float32)
        self.Dropout_Rate1 = tf.placeholder(tf.float32)
        self.Dropout_Rate2 = tf.placeholder(tf.float32)
        self.TRAIN_PH = tf.placeholder(tf.bool)
        self.jaso_Onehot = self.JM.string_to_index(self.X_Onehot)


    ########################################################
    # Network structure
    ########################################################
    def set_network(self):

        ##########################################
        # Recurrent layer
        ##########################################
        with tf.variable_scope('LSTMcell'):
            # 여러개의 셀을 조합한 RNN 셀을 생성합니다.
            multi_cells = tf.contrib.rnn.MultiRNNCell(
                [utils.LSTM_cell(FLAGS.RNN_HIDDEN_DIMENSION, self.Dropout_Rate1, self.Dropout_Rate2) for _ in range(FLAGS.N_LAYERS)])

            # RNN 신경망을 생성 (SEQ를 통해 매 sentence의 길이까지만 계산을 해 효율성 증대)
            self.outputs, _states = tf.nn.dynamic_rnn(cell=multi_cells, inputs=self.X, dtype=tf.float32)  # sequence_length=self.SEQ,

            # 마지막에 해당하는 결과물을 가져옴 (utils의 last_relevant 함수를 이용)
            self.rnn_outputs = utils.last_relevant(self.outputs, self.SEQ)


        ##########################################
        # Fully connected network
        ##########################################
        with tf.variable_scope('FC-layer'):
            FC1 = tf.contrib.layers.fully_connected(self.rnn_outputs, FLAGS.FC_HIDDEN_DIMENSION, activation_fn=None)
            FC_act1 = tf.nn.relu(tf.layers.batch_normalization(FC1, momentum=0.9, training=self.TRAIN_PH))
            FC2 = tf.contrib.layers.fully_connected(FC_act1, FLAGS.FC_HIDDEN_DIMENSION, activation_fn=None)
            FC_act2 = tf.nn.relu(tf.layers.batch_normalization(FC2, momentum=0.9, training=self.TRAIN_PH))

            self.y_logits = tf.contrib.layers.fully_connected(FC_act2, FLAGS.NUM_OF_CLASS, activation_fn=None)



    ########################################################
    # Optimizer / accuracy / loss / tensorboard
    ########################################################
    def set_ops(self, WRITER, WRITER_generate):
        # loss
        with tf.variable_scope('cross_entropy'):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits, labels=self.Y, name='loss'))
            tf.summary.scalar('cross_entropy', self.cross_entropy)

        # optimizer
        with tf.variable_scope('Optimizer'):
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.optm = tf.train.RMSPropOptimizer(self.LEARNING_RATE).minimize(loss=self.cross_entropy)

        # accuracy
        with tf.variable_scope('accuracy'):
            with tf.variable_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.y_logits, 1), tf.argmax(self.Y, 1))
            with tf.variable_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc')
                tf.summary.scalar('accuracy', self.accuracy)
        print("Optimizer Ready! & Let's Train!")


        self.merge = tf.summary.merge_all()
        if WRITER_generate:
            self.train_writer = tf.summary.FileWriter("./output/{}/train_{}".format(WRITER, WRITER), self.sess.graph)
            self.test_writer = tf.summary.FileWriter("./output/{}/test_{}".format(WRITER, WRITER), self.sess.graph)
        print("Graph Ready!")

