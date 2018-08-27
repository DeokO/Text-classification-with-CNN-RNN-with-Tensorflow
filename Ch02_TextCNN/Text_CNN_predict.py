####################################################
# Text classification with CNN - predict (character level)
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.05.
####################################################



#######################################################################
### For escape tensorflow early stop error ( CTRL + C )
#######################################################################
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
#####################################
# Import modules
#####################################
from Ch01_Data_load import data_load, utils
from Ch01_Data_load import Jaso_mapping_utils as jmu
from Ch02_TextCNN.Text_CNN_config import *
from Ch02_TextCNN.Text_CNN_model import *



################################################################################
# DATA LOAD
################################################################################
TRAIN_DOC, TRAIN_LABEL, TRAIN_LABEL_POS, TRAIN_LABEL_NEG, TEST_DOC, TEST_LABEL, TEST_LABEL_POS, TEST_LABEL_NEG = data_load.data_load()
JM = utils.lookup_JM(FLAGS.INPUT_WIDTH, FLAGS.INPUT_DEPTH)
FLAGS.WRITER_generate = False



################################################################################
# Start Session / Network Scratch / Restore Check Point
################################################################################
# Start Session
sess = tf.Session()
print("Session Ready!")
model = MODEL(sess=sess, JM=JM, FLAGS=FLAGS)

# Initialization
sess.run(tf.global_variables_initializer())
model.JM.init_table(sess)

# Restore parameter
saver = tf.train.Saver()
saver.restore(sess, "./Saver/{}/{}.ckpt".format(FLAGS.WRITER, FLAGS.WRITER))



################################################################################
# Get performance scores
################################################################################
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Calculate logits
LOGIT_list = np.empty([0, 2])
LABEL_list = np.empty([0, 2])
for i in range(int(len(TEST_DOC) / FLAGS.TEST_BATCH)+1):
    index = np.unique(np.clip(np.arange(i * FLAGS.TEST_BATCH, (i + 1) * FLAGS.TEST_BATCH), a_min=0, a_max=len(TEST_DOC) - 1))
    jaso_splitted = jmu.jaso_split(TEST_DOC[index], MAXLEN=FLAGS.INPUT_WIDTH)
    batch_input = sess.run(model.jaso_Onehot, {model.X_Onehot: jaso_splitted})
    batch_label = TEST_LABEL[index]
    ts_acc, y_logit = sess.run([model.accuracy, model.y_logits],
                               feed_dict={model.X: batch_input,
                                          model.Y: batch_label,
                                          model.TRAIN_PH: False})
    LOGIT_list = np.concatenate([LOGIT_list, y_logit])
    LABEL_list = np.concatenate([LABEL_list, batch_label])
    print(i, '||', ts_acc)

# Calculate AUROC, accuracy from confusion matrix
softmax_logit = np.array(list(map(lambda x: np.exp(x) / sum(np.exp(x)), LOGIT_list)))
y_true = LABEL_list[:, 1]
y_pred = np.argmax(softmax_logit, axis=1)
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
cm = confusion_matrix(y_true, y_pred)

print(cm)
print('AUROC: {},        acc: {} '.format(metrics.auc(fpr, tpr), (cm[0, 0]+cm[1, 1]) / np.sum(cm)))



# [[37796  4021]
#  [ 3963 28769]]
# AUROC: 0.8913843782586843,        acc: 0.8929026546298408
