####################################################
# Jaso wise split & mapping utils
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.01.
####################################################



#####################################
# Import modules
#####################################
import tensorflow as tf
import numpy as np
# from hangul_utils import split_syllables #설치 : https://github.com/kaniblu/hangul-utils

import six

def __c(x):
    return six.unichr(x)

INITIALS = list(map(__c, [0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
                          0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
                          0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
                          0x314e]))

MEDIALS = list(map(__c, [0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
                         0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
                         0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
                         0x3161, 0x3162, 0x3163]))

FINALS = list(map(__c, [0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
                        0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
                        0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
                        0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
                        0x314c, 0x314d, 0x314e]))

def check_syllable(x):
    return 0xAC00 <= ord(x) <= 0xD7A3

def split_syllable_char(x):
    """
    Splits a given korean character into components.
    :param x: A complete korean character.
    :return: A tuple of basic characters that constitutes the given characters.
    """
    if len(x) != 1:
        raise ValueError("Input string must have exactly one character.")

    if not check_syllable(x):
        raise ValueError(
            "Input string does not contain a valid Korean character.")

    diff = ord(x) - 0xAC00
    _m = diff % 28
    _d = (diff - _m) // 28

    initial_index = _d // 21
    medial_index = _d % 21
    final_index = _m

    if not final_index:
        result = (INITIALS[initial_index], MEDIALS[medial_index])
    else:
        result = (
            INITIALS[initial_index], MEDIALS[medial_index],
            FINALS[final_index - 1])

    return result

def split_syllables(string):
    """
    Splits a sequence of Korean syllables to produce a sequence of jamos.
    Irrelevant characters will be ignored.
    :param string: A unicode string.
    :return: A converted unicode string.
    """
    new_string = ""
    for c in string:
        if not check_syllable(c):
            new_c = c
        else:
            new_c = "".join(split_syllable_char(c))
        new_string += new_c

    return new_string


###############################################################
# jaso mapping class
###############################################################
#Jaso split function
def jaso_split(DOCUMENT, MAXLEN=256): # 자소로 나눈 다음 95분위수에 해당하는 개수 166 보다 큰 256으로 설정
    doc = DOCUMENT
    JASO_RESULT = []
    # lens = []
    for i in np.arange(len(doc)):

        jamo = split_syllables(doc[i])
        jamo = list(jamo.replace('  ', ' '))
        length = len(jamo)

        if length < MAXLEN:
            diff = MAXLEN - length
            jamo.extend(np.repeat('ⓟ', diff))
        else:
            jamo = jamo[:MAXLEN]

        JASO_RESULT.append(jamo)
        # lens.append(length)

    return JASO_RESULT #, lens



class JasoMapping():
    #생성자
    def __init__(self, WIDTH, DEPTH, MAPPING_KEY, MAPPING_VALUE):
        self.WIDTH = WIDTH
        self.DEPTH = DEPTH
        self.MAPPING_KEY = MAPPING_KEY
        self.MAPPING_VALUE = MAPPING_VALUE

        #define jaso to index mapping table
        self.table = tf.contrib.lookup.HashTable(
                            tf.contrib.lookup.KeyValueTensorInitializer(keys=self.MAPPING_KEY, values=self.MAPPING_VALUE),
                            default_value=118)

    #initialize table
    def init_table(self, sess):
        sess.run(tf.tables_initializer())

    def one_hot(self, index):
        onehot = tf.one_hot(indices=index, depth=self.DEPTH)
        return tf.reshape(onehot, [-1, self.WIDTH, self.DEPTH])

    #Jaso mapping to index
    def string_to_index(self, STRING_DATA, num_oov_buckets=1, default_value=118):
        self.indices = self.table.lookup(STRING_DATA)
        return self.one_hot(self.indices)





# # 사용법
# ###############################################################
# #HYPER PARAMETER
# ###############################################################
# WIDTH = 256
# DEPTH = 93
#
# ###############################################################
# #DATA LOAD
# ###############################################################
# import pandas as pd
# DATA_PATH = './Ch01_Data_load/data/w_movie.csv'
# data = pd.read_csv(DATA_PATH, encoding='cp949')
# DOCUMENT = data.review_text
# POINT = data.review_point
#
# #lookup table dictionary
# MAPPING_PATH = './Ch01_Data_load/data/dict.csv'
# lookup = pd.read_csv(MAPPING_PATH, encoding='cp949')
# keys = list(lookup.iloc[:, 0])
# values = list(lookup.iloc[:, 1])
#
#
# JM = JasoMapping(WIDTH=WIDTH, DEPTH=DEPTH, MAPPING_KEY=keys, MAPPING_VALUE=values)
# X = tf.placeholder(dtype=tf.string, shape=[None, WIDTH])
#
# splited = jaso_split(DOCUMENT[:128], MAXLEN=WIDTH)
# np.shape(splited)
# print(splited[0])
# sess = tf.Session()
# JM.init_table(sess)
# INDEX_onehot = JM.string_to_index(X)
# tmpv = JM.indices
# tmpv_run = sess.run(tmpv, {X: splited})
# np.shape(tmpv_run)
# tmp = sess.run(INDEX_onehot, {X: splited})
# np.shape(tmp)
