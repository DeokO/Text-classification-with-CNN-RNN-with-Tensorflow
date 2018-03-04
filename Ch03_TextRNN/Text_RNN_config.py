####################################################
# Text classification with RNN - config (character level)
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.07.
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
flags.DEFINE_integer('INPUT_WIDTH', 256, 'document size')
flags.DEFINE_integer('INPUT_DEPTH', 93, 'The number of terms')
flags.DEFINE_integer('MAXLEN', 256, 'max length of document')

# Class
flags.DEFINE_integer('NUM_OF_CLASS', 2, 'positive, negative')

# Parameter
flags.DEFINE_integer('RNN_HIDDEN_DIMENSION', 256, 'hidden dimension')
flags.DEFINE_integer('FC_HIDDEN_DIMENSION', 256, 'hidden dimension')
flags.DEFINE_float('Dropout_Rate1', 0.8, 'Dropout_Rate1')
flags.DEFINE_float('Dropout_Rate2', 0.8, 'Dropout_Rate2')
flags.DEFINE_integer('N_LAYERS', 3, 'The number of layers')

# Save
flags.DEFINE_string('WRITER', 'Text_RNN', 'saver name')
flags.DEFINE_boolean('WRITER_generate', True, 'saver generate')

# Train
flags.DEFINE_integer('BATCH_SIZE', 128, 'batch size')
flags.DEFINE_integer('TEST_BATCH', 128, 'test batch size')
flags.DEFINE_integer('NUM_OF_EPOCH', 50, 'number of epoch')
flags.DEFINE_float('lr_value', 0.1, 'initial learning rate')
flags.DEFINE_float('lr_decay', 0.8, 'learning rate decay')
flags.DEFINE_integer('Check_Loss', np.repeat(5, 20), 'loss decay')

# FLAGS
FLAGS = flags.FLAGS
FLAGS._parse_flags()
