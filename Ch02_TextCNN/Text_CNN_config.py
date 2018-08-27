####################################################
# Text classification with CNN - config (character level)
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.05.
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
flags.DEFINE_integer('INPUT_DEPTH', 93, 'The number of terms')
flags.DEFINE_integer('INPUT_WIDTH', 256, 'max length of document')

# Class
flags.DEFINE_integer('NUM_OF_CLASS', 2, 'positive, negative')

# Parameter
flags.DEFINE_integer('HIDDEN_DIMENSION', 256, 'hidden dimension')
flags.DEFINE_list('CONV_KERNEL_WIDTH', [19, 13], 'kernel height')

# Save
flags.DEFINE_string('WRITER', 'Text_CNN', 'saver name')
flags.DEFINE_boolean('WRITER_generate', True, 'saver generate')

# Train
flags.DEFINE_integer('BATCH_SIZE', 128, 'batch size')
flags.DEFINE_integer('TEST_BATCH', 128, 'test batch size')
flags.DEFINE_integer('NUM_OF_EPOCH', 3, 'number of epoch')
flags.DEFINE_float('lr_value', 0.01, 'initial learning rate')
flags.DEFINE_float('lr_decay', 0.5, 'learning rate decay')
flags.DEFINE_multi_integer('Check_Loss', [5]*20, 'loss decay')

# FLAGS
FLAGS = flags.FLAGS


