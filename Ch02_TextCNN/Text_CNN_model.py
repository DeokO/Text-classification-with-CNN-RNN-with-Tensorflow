####################################################
# Text classification with CNN - model (character level)
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.05.
####################################################



#####################################
# Import modules
#####################################
import gc
gc.collect()
import tensorflow as tf
from Ch02_TextCNN.Text_CNN_config import *
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
        self.LEARNING_RATE = tf.placeholder(tf.float32)
        self.TRAIN_PH = tf.placeholder(tf.bool)
        self.jaso_Onehot = self.JM.string_to_index(self.X_Onehot)


    ########################################################
    # Network structure
    ########################################################
    def set_network(self):

        ##########################################
        # Convolutional layer
        ##########################################
        net = self.X
        for i, width in enumerate(FLAGS.CONV_KERNEL_WIDTH):
            with tf.variable_scope('conv_{}'.format(i+1)):
                # 1d-conv
                net = utils.conv_1d(net, width, FLAGS.HIDDEN_DIMENSION, self.TRAIN_PH)
                # 1d-max pool
                if i % 2 == 1:
                    net = tf.layers.max_pooling1d(net, pool_size=2, strides=2, padding='SAME')


        ##########################################
        # Reshape
        ##########################################
        self.Activation = net
        self.GAP = tf.layers.average_pooling1d(self.Activation, pool_size=self.Activation.get_shape().as_list()[1],
                                               strides=self.Activation.get_shape().as_list()[1], padding='SAME')

        ##########################################
        # Fully connected network
        ##########################################
        with tf.variable_scope('FC_Network_sentiment'):
            FC = tf.layers.dense(inputs=self.GAP, units=FLAGS.NUM_OF_CLASS, kernel_initializer=utils.he_init,
                                 bias_initializer=tf.zeros_initializer())
            FC_bn = tf.layers.batch_normalization(FC, momentum=0.9, training=self.TRAIN_PH)
            self.y_logits = tf.reshape(FC_bn, [-1, FLAGS.NUM_OF_CLASS])


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
                self.optm = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(loss=self.cross_entropy)

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
