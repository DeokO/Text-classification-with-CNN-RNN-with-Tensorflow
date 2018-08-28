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









# # 예시용 코드 (출처: https://stackoverflow.com/questions/40661684/tensorflow-vocabularyprocessor)
# import numpy as np
# from tensorflow.contrib import learn
#
# x_text = ['This is a cat','This must be boy', 'This is a a dog']
# max_document_length = max([len(x.split(" ")) for x in x_text])
#
# ## Create the vocabularyprocessor object, setting the max lengh of the documents.
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#
# ## Transform the documents using the vocabulary.
# x = np.array(list(vocab_processor.fit_transform(x_text)))
#
# ## Extract word:id mapping from the object.
# vocab_dict = vocab_processor.vocabulary_._mapping
#
# ## Sort the vocabulary dictionary on the basis of values(id).
# ## Both statements perform same task.
# #sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
# sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
#
# ## Treat the id's as index into list and create a list of words in the ascending order of id's
# ## word with id i goes at index i of the list.
# vocabulary = list(list(zip(*sorted_vocab))[0])
#
# print(vocabulary)
# print(x)
