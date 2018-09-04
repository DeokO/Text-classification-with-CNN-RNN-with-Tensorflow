####################################################
# Utils
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.01.
####################################################



#####################################
# Import modules
#####################################
import tensorflow as tf
import pandas as pd
import numpy as np
from soynlp.tokenizer import RegexTokenizer #https://github.com/lovit/soynlp

import Ch01_Data_load.Jaso_mapping_utils as jmu

# initializer
he_init = tf.contrib.layers.variance_scaling_initializer()



################################################################################
# Define layers
################################################################################
def conv_1d(DATA, KERNEL_WIDTH, DEPTH, TRAIN_PH):
    #이전 layer에서 온 conv의 feature map의 in_depth
    in_depth = DATA.get_shape().as_list()[2]
    #weight variable 설정
    Weight = tf.get_variable(name='weight', shape=[KERNEL_WIDTH, in_depth, DEPTH], initializer=he_init)
    conv = tf.nn.conv1d(value=DATA, filters=Weight, stride=1, padding='SAME')
    bn = tf.layers.batch_normalization(conv, momentum=0.9, training=TRAIN_PH)
    relu = tf.nn.relu(bn)
    return relu



# RNN 에 학습에 사용할 셀을 생성
# BasicRNNCell, BasicLSTMCell, GRUCell 들을 사용하면 다른 구조의 셀로 간단하게 변경 가능하며
# 본 코드에서는 gru를 이용하고, 과적합 방지를 위해 dropout을 적용해 주었음
def GRU_cell(n_hidden, Dropout_Rate1, Dropout_Rate2):
    cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)
    return cell
def LSTM_cell(n_hidden, Dropout_Rate1, Dropout_Rate2):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)
    return cell
def RNN_cell(n_hidden, Dropout_Rate1, Dropout_Rate2):
    cell = tf.contrib.rnn.BasicRNNCell(num_units=n_hidden)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)
    return cell

def RNN_structure(cell_type, n_hidden, Dropout_Rate1, Dropout_Rate2, n_layers):
    if cell_type == 'GRU':
        # 여러개의 셀을 조합한 RNN 셀을 생성합니다.
        multi_cells = tf.contrib.rnn.MultiRNNCell(
            [GRU_cell(n_hidden, Dropout_Rate1, Dropout_Rate2) for _ in range(n_layers)])
    elif cell_type == 'LSTM':
        multi_cells = tf.contrib.rnn.MultiRNNCell(
            [LSTM_cell(n_hidden, Dropout_Rate1, Dropout_Rate2) for _ in range(n_layers)])
    elif cell_type == 'RNN':
        multi_cells = tf.contrib.rnn.MultiRNNCell(
            [RNN_cell(n_hidden, Dropout_Rate1, Dropout_Rate2) for _ in range(n_layers)])
    else:
        return None
    return multi_cells



################################################################################
# dynamic_rnn에서 할당한 sequence_length에 대해 해당 위치의 output을 뽑아오는 함수
################################################################################
def last_relevant(outputs, seq_length):
    with tf.variable_scope('StepSelector'):
        #batch_size
        batch_size = tf.shape(outputs)[0]
        #max_length
        max_length = tf.shape(outputs)[1]
        #out_size: hidden vector 길이
        out_size = int(outputs.get_shape()[2])

        # rnn을 지나며 산출된 output에 대해 flatten 진행
        flat = tf.reshape(outputs, [-1, out_size])

        #어떤 부분이 마지막 산출물인지 index 산출
        index = tf.range(0, batch_size) * max_length + (seq_length-1)
        #먼저 one-hot으로 해당 위치만 1을 찍은 벡터들을 모두 만든 다음, 행 방향으로 모두 더해서 하나의 벡터로 partitions 벡터를 만듦
        partitions = tf.reduce_sum(tf.one_hot(index, tf.shape(flat)[0], dtype='int32'), 0)

        # 위에서 만든 partitions로 2개의 파티션을 flat에 대해 만들어 주는데, 0 그룹과 1 그룹이 생기게 됨
        last_timesteps = tf.dynamic_partition(data=flat, partitions=partitions, num_partitions=2)  # (batch_size, n_dim)
        # 1에 해당하는 partition들이 관심을 가지고 볼 각 input에 대해 제일 마지막의 rnn 결과물이므로 1 그룹을 선택해 줌
        last_timesteps = last_timesteps[1]

    return last_timesteps



################################################################################
# input array에서 패딩이 위치한 곳 찾는 함수
################################################################################
def find_length(sequence):
    try:
        leng = sequence.index('ⓟ')
        return leng
    except ValueError as e:
        leng = len(sequence)
        return leng



################################################################################
# dynamic_rnn에서 sequence_length 옵션에 전달할 각 문서의 길이를 산출하는 함수
################################################################################
def length(batch_x):
    non_unk = [x > 0 for x in batch_x]
    leng = []
    del_list = []
    for idx, token in enumerate(non_unk):
        tmp = np.where(token)[0]
        if len(tmp) != 0:
            if tmp[-1] == 0:
                del_list.append(idx)
            else:
                leng.append(tmp[-1])
        else:
            del_list.append(idx)

    return np.array(leng), del_list



################################################################################
# batch generate
################################################################################
def sampler(LABEL_POS, LABEL_NEG, BATCH_SIZE):
    pos = np.random.choice(LABEL_POS, int(BATCH_SIZE / 2), replace=False)
    neg = np.random.choice(LABEL_NEG, int(BATCH_SIZE / 2), replace=False)
    return np.r_[pos, neg]


def generate_batch_jaso(INDEX, MODEL, DOC, LABEL, MAXLEN, SESS):
    jaso_splitted = jmu.jaso_split(DOC[INDEX], MAXLEN=MAXLEN)
    _input = SESS.run(MODEL.jaso_Onehot, {MODEL.X_Onehot: jaso_splitted})
    _, del_list = length(_input)
    _label = LABEL[INDEX]
    batch_input = np.delete(_input, del_list, axis=0)
    batch_label = np.delete(_label, del_list, axis=0)
    return batch_input, batch_label


def generate_batch_word(INDEX, VOCAB_PROCESSOR, DOC, LABEL):
    _input = np.array(list(VOCAB_PROCESSOR.transform(DOC[INDEX])))
    _, del_list = length(_input)
    _label = LABEL[INDEX]
    batch_input = np.delete(_input, del_list, axis=0)
    batch_label = np.delete(_label, del_list, axis=0)

    return batch_input, batch_label



################################################################################
# CAM interpolate
################################################################################
def double_interpolate(score):
    N = len(score)
    X = np.arange(0, 2 * N, 2)
    X_new = np.sort(np.r_[X, X+1])
    score_new = np.interp(X_new, X, score)
    return score_new



################################################################################
# Min Max regularizer
################################################################################
def MINMAX(Data):
    return ((Data - np.min(Data)) / (np.max(Data) - np.min(Data)))



################################################################################
# lookup table for jaso one-hot
################################################################################
def lookup_JM(WIDTH, DEPTH):
    MAPPING_PATH = './Ch01_Data_load/data/dict.csv'
    lookup = pd.read_csv(MAPPING_PATH, encoding='cp949')
    keys = list(lookup.iloc[:, 0])
    values = list(lookup.iloc[:, 1])
    JM = jmu.JasoMapping(WIDTH=WIDTH, DEPTH=DEPTH, MAPPING_KEY=keys, MAPPING_VALUE=values)
    return JM



################################################################################
# Tokenize
################################################################################
def Tokenize(data):
    tokenizer = RegexTokenizer()
    output = list(map(lambda x: ' '.join(tokenizer.tokenize(x)), data))
    return output



################################################################################
# RNN attention
################################################################################
def attention(INPUTS, ATTENTION_SIZE, SEQ, time_major=False, return_alphas=False):

    if isinstance(INPUTS, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        INPUTS = tf.concat(INPUTS, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        INPUTS = tf.array_ops.transpose(INPUTS, [1, 0, 2])

    inputs_shape = INPUTS.shape
    hidden_size = inputs_shape[2].value

    # Attention mechanism
    W_omega = tf.get_variable(name='W_omega', shape=[hidden_size, ATTENTION_SIZE], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(stddev=0.1))
    b_omega = tf.get_variable(name='b_omega', shape=[ATTENTION_SIZE], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(stddev=0.1))
    u_omega = tf.get_variable(name='u_omega', shape=[ATTENTION_SIZE], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(INPUTS, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.reshape(tf.matmul(v, tf.reshape(u_omega, [-1, 1])), [-1, inputs_shape[1].value])
    alphas = tf.nn.softmax(vu)

    output = tf.reduce_sum(INPUTS * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
