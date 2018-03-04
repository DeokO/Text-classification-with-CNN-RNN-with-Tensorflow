####################################################
# Word wise split & mapping utils
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.09.
####################################################



#####################################
# Import modules
#####################################
import tensorflow as tf
import numpy as np
from Ch01_Data_load import data_load



##########################
# max length of document
##########################
TRAIN_DOC, TRAIN_LABEL, _, _, TEST_DOC, TEST_LABEL, _, _ = data_load.data_load()
max_doc_length = max([len(x.split(" ")) for x in TRAIN_DOC])



##########################
# Vocabulary Processor
##########################
# Create the vocabularyprocessor object, setting the max lengh of the documents.
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_doc_length, min_frequency=5)

# Transform the documents using the vocabulary.
x = np.array(list(vocab_processor.fit_transform(TRAIN_DOC)))

## Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping
# len(vocab_dict) #72844

# save vocabularyprocessor object
vocab_processor.save('./Ch01_Data_load/data/VocabularyProcessor')
