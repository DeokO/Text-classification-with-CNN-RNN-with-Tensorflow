####################################################
# Text classification with attention RNN - predict (word level)
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.19.
####################################################



#######################################################################
### For escape tensorflow early stop error ( CTRL + C )
#######################################################################
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
#####################################
# Import modules
#####################################
from Ch01_Data_load import data_load
from Ch05_TextRNN_word_attention.Text_RNN_word_attention_model import *



################################################################################
# DATA LOAD
################################################################################
TRAIN_DOC, TRAIN_LABEL, TRAIN_LABEL_POS, TRAIN_LABEL_NEG, TEST_DOC, TEST_LABEL, TEST_LABEL_POS, TEST_LABEL_NEG = data_load.data_load()

# restore vocabularyprocessor object
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(FLAGS.Vocab_Processor_PATH)

FLAGS.WRITER_generate = False



################################################################################
# Start Session / Network Scratch / Restore Check Point
################################################################################
# Start Session
sess = tf.Session()
print("Session Ready!")
model = MODEL(sess=sess, FLAGS=FLAGS)

# Initialization
sess.run(tf.global_variables_initializer())

# Restore parameter
saver = tf.train.Saver()
saver.restore(sess, "./Saver/{}/{}.ckpt".format(FLAGS.WRITER, FLAGS.WRITER))



################################################################################
# Save parameters
################################################################################
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Calculate logits
LOGIT_list = np.empty([0, 2])
LABEL_list = np.empty([0, 2])
for i in range(int(len(TEST_DOC) / FLAGS.TEST_BATCH)+1):

    index = np.unique(np.clip(np.arange(i*FLAGS.TEST_BATCH, (i+1)*FLAGS.TEST_BATCH), a_min=0, a_max=len(TEST_DOC)-1))
    batch_input, batch_label = utils.generate_batch_word(INDEX=index, VOCAB_PROCESSOR=vocab_processor,
                                                         DOC=TEST_DOC, LABEL=TEST_LABEL, ATTENTION=False)
    seq_length, _ = utils.length(batch_input)

    ts_acc, y_logit = sess.run([model.accuracy, model.y_logits],
                               feed_dict={model.X_idx: batch_input,
                                          model.Y: batch_label,
                                          model.LEARNING_RATE: FLAGS.lr_value,
                                          model.SEQ: seq_length,
                                          model.Dropout_Rate1: 1,
                                          model.Dropout_Rate2: 1,
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



# [[72891 19685]
#  [ 9688 44657]]
# AUROC: 0.8045477128248277,        acc: 0.8000762314441094
