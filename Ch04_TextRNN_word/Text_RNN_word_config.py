####################################################
# Text classification with RNN - model (word level)
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.19.
####################################################



#######################################################################
### Import modules
#######################################################################
import tensorflow as tf
from tensorflow import flags
import numpy as np


#######################################################################
### Hyper parameter setting
#######################################################################
# Input data
flags.DEFINE_string('Vocab_Processor_PATH', './Ch01_Data_load/data/VocabularyProcessor', 'VocabularyProcessor object file path')
flags.DEFINE_integer('VOCAB_SIZE', 72844, 'The number of terms in vocabulary')
flags.DEFINE_integer('EMBEDDING_SIZE', 256, 'Dimension of embedded terms')
flags.DEFINE_integer('MAXLEN', 22, 'max length of document')

# Class
flags.DEFINE_integer('NUM_OF_CLASS', 2, 'positive, negative')

# Parameter
flags.DEFINE_string('RNN_CELL', 'LSTM', 'RNN cell default LSTM')
flags.DEFINE_integer('RNN_HIDDEN_DIMENSION', 256, 'RNN hidden dimension')
flags.DEFINE_integer('FC_HIDDEN_DIMENSION', 256, 'FC hidden dimension')
flags.DEFINE_float('Dropout_Rate1', 0.7, 'Dropout_Rate1')
flags.DEFINE_float('Dropout_Rate2', 0.7, 'Dropout_Rate2')
flags.DEFINE_integer('N_LAYERS', 3, 'The number of layers')

# Save
flags.DEFINE_string('WRITER', 'Text_RNN_word', 'saver name')
flags.DEFINE_boolean('WRITER_generate', True, 'saver generate')

# Train
flags.DEFINE_integer('BATCH_SIZE', 128, 'batch size')
flags.DEFINE_integer('TEST_BATCH', 128, 'test batch size')
flags.DEFINE_integer('NUM_OF_EPOCH', 3, 'number of epoch')
flags.DEFINE_float('lr_value', 0.01, 'initial learning rate')
flags.DEFINE_float('lr_decay', 0.7, 'learning rate decay')
flags.DEFINE_multi_integer('Check_Loss', [5]*20, 'loss decay')

# FLAGS
FLAGS = flags.FLAGS


