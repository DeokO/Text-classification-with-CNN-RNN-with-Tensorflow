####################################################
# Text classification with attention RNN - train (word level)
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
from Ch05_TextRNN_word_attention.Text_RNN_word_attention_config import *
from Ch05_TextRNN_word_attention.Text_RNN_word_attention_model import *
information = ''
FLAGS.WRITER += information



################################################################################
# DATA LOAD
################################################################################
TRAIN_DOC, TRAIN_LABEL, TRAIN_LABEL_POS, TRAIN_LABEL_NEG, TEST_DOC, TEST_LABEL, TEST_LABEL_POS, TEST_LABEL_NEG = data_load.data_load()

# restore vocabularyprocessor object
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(FLAGS.Vocab_Processor_PATH)



################################################################################
# Start Session / Network Scratch / Save Check Point
################################################################################
# Start Session
sess = tf.Session()
print("Session Ready!")
model = MODEL(sess=sess, FLAGS=FLAGS)

# Initialization
sess.run(tf.global_variables_initializer())



################################################################################
# Let's Train!!
################################################################################
# 한 epoch 당 iteration 횟수
Num_of_Iterlation = np.shape(TRAIN_DOC)[0] // FLAGS.BATCH_SIZE
epoch = 0
for i in range(Num_of_Iterlation * FLAGS.NUM_OF_EPOCH):
    # i=0
    if i % Num_of_Iterlation == 0:
        epoch += 1
        FLAGS.lr_value *= FLAGS.lr_decay

    ################################################################
    # Training batch OPTIMIZE
    ################################################################
    index = utils.sampler(LABEL_POS=TRAIN_LABEL_POS, LABEL_NEG=TRAIN_LABEL_NEG, BATCH_SIZE=FLAGS.BATCH_SIZE)
    batch_input, batch_label = utils.generate_batch_word(INDEX=index, VOCAB_PROCESSOR=vocab_processor,
                                                         DOC=TRAIN_DOC, LABEL=TRAIN_LABEL)
    seq_length, _ = utils.length(batch_input)

    _ = sess.run([model.optm],
                 feed_dict={model.X_idx: batch_input,
                            model.Y: batch_label,
                            model.LEARNING_RATE: FLAGS.lr_value,
                            model.SEQ: seq_length,
                            model.Dropout_Rate1: FLAGS.Dropout_Rate1,
                            model.Dropout_Rate2: FLAGS.Dropout_Rate2,
                            model.TRAIN_PH: True})


    ################################################################
    # Calculate Train & Test Loss, Accuracy & Summary / Print
    ################################################################
    if i % 10 == 0:
        ################################################################
        # Train batch LOSS CHECK
        ################################################################
        tr_loss, tr_acc, tr_merged = sess.run([model.cross_entropy, model.accuracy, model.merge],
                                              feed_dict={model.X_idx: batch_input,
                                                         model.Y: batch_label,
                                                         model.LEARNING_RATE: FLAGS.lr_value,
                                                         model.SEQ: seq_length,
                                                         model.Dropout_Rate1: 1,
                                                         model.Dropout_Rate2: 1,
                                                         model.TRAIN_PH: True})
        model.train_writer.add_summary(tr_merged, i)


        ################################################################
        # Test batch LOSS CHECK
        ################################################################
        index = utils.sampler(LABEL_POS=TEST_LABEL_POS, LABEL_NEG=TEST_LABEL_NEG, BATCH_SIZE=FLAGS.TEST_BATCH)
        batch_input, batch_label = utils.generate_batch_word(INDEX=index, VOCAB_PROCESSOR=vocab_processor,
                                                             DOC=TEST_DOC, LABEL=TEST_LABEL)
        seq_length, _ = utils.length(batch_input)

        ts_loss, ts_acc, ts_merged = sess.run([model.cross_entropy, model.accuracy, model.merge],
                                              feed_dict={model.X_idx: batch_input,
                                                         model.Y: batch_label,
                                                         model.LEARNING_RATE: FLAGS.lr_value,
                                                         model.SEQ: seq_length,
                                                         model.Dropout_Rate1: 1,
                                                         model.Dropout_Rate2: 1,
                                                         model.TRAIN_PH: False})
        model.test_writer.add_summary(ts_merged, i)

        ################################################################
        # Print
        ################################################################
        print("Iter: {iter:08} / Epoch: {EP} |##| LR: {LR:0.15f} |##|  tr_LOSS: {tr_LOSS:0.8f} |##|  tr_acc: {tr_ACC:0.8f} |##|  ts_LOSS: {ts_loss:0.8f} |##|  ts_acc: {ts_acc:0.8f}".format(
                iter=i, EP=epoch, LR=FLAGS.lr_value, tr_LOSS=tr_loss, tr_ACC=tr_acc, ts_loss=ts_loss, ts_acc=ts_acc))
        FLAGS.Check_Loss = FLAGS.Check_Loss[1:]
        FLAGS.Check_Loss.append(round(tr_loss, 4))
        print(FLAGS.Check_Loss)


    ##############################################################################################################
    ##############################################################################################################



################################################################################
# Save parameters
################################################################################
# Save Weights
if FLAGS.WRITER not in os.listdir("./Saver"):
    os.makedirs("./Saver/{}".format(FLAGS.WRITER))
saver = tf.train.Saver()
saver.save(sess, "./Saver/{}/{}.ckpt".format(FLAGS.WRITER, FLAGS.WRITER))


