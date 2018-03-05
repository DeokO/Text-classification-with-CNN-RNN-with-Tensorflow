# Text-classification-with-CNN-RNN
Text classification with CNN, RNN, RCNN model by character, word level

-----------------------------------------------------------
+ 한글 문서에 대해 classifier를 만든 예
+ Tensorflow를 이용했으며, Convolutional layer, Recurrent layer를 이용
+ 실 데이터를 사용(전에 공부용으로 수집한 데이터 인데, 문제될 경우 내리겠습니다.)

## Ch01_Data_load
***data link: https://drive.google.com/open?id=1vdvedBSAcVU8Dbjzuow6cL_9Tkc9zGVM***
-출처: W 영화 사이트
1. [data_preprocessing.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch01_Data_load/data_preprocessing.py): 텍스트 기본 전처리 진행
    - 특수문자 제거
    - corpus의 문장 길이 10분위수 ~ 95분위수만 사용
    - 2점 이하: 부정, 5점: 긍정으로 나누어 label 비율을 최대한 반반으로 설정
2. [data_load.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch01_Data_load/data_load.py): 데이터 불러오기
3. [Jaso_mapping_utils.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch01_Data_load/Jaso_mapping_utils.py): 텐서에서 자소를 onehot vector로 변환
    - 자소 단위로 input을 받는 모형에 대해 적용
4. [make_VocabularyProcessor.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch01_Data_load/make_VocabularyProcessor.py): 텐서에서 단어를 index로 변환
    - 제공하는 VocabularyProcessor를 사용하여 객체 생성
5. [utils.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch01_Data_load/utils.py): layers, batch generate, tokenizer 등 사용할 함수 정의

## Ch02_TextCNN
***character level CNN text classifier***
+ **자소 단위**의 input을 받는 **CNN** text classifier
+ 인터넷의 언어 파괴적 문서, 오타 등에 강건한 모형
1. [Text_CNN_config.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch02_TextCNN/Text_CNN_config.py):
    - model과 관련한 hyper-parameter 정의
2. [Text_CNN_model.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch02_TextCNN/Text_CNN_model.py):
    - model class
3. [Text_CNN_train.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch02_TextCNN/Text_CNN_train.py):
    - 모형을 학습하고, tensorboard로 summary를 확인 및 학습된 파라미터 저장
4. [Text_CNN_predict.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch02_TextCNN/Text_CNN_train.py):
    - 학습된 모형에 test data를 적용해 성능지표 산출

## Ch03_TextRNN
***character level RNN text classifier***
+ **자소 단위**의 input을 받는 **RNN** text classifier
+ 불필요하게 sequence를 길게한 탓인지, 자소 단위의 RNN은 학습이 잘 되지 않음
+ (이에 대한 원인을 알고 있으신 분이 있으시면 메일 부탁드립니다.)
1. [Text_RNN_config.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch03_TextRNN/Text_RNN_config.py):
    - model과 관련한 hyper-parameter 정의
2. [Text_RNN_model.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch03_TextRNN/Text_RNN_model.py):
    - model class
3. [Text_RNN_train.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch03_TextRNN/Text_RNN_train.py):
    - 모형을 학습하고, tensorboard로 summary를 확인 및 학습된 파라미터 저장
4. [Text_RNN_predict.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch03_TextRNN/Text_RNN_train.py):
    - 학습된 모형에 test data를 적용해 성능지표 산출

## Ch04_TextRNN_word
***word level RNN text classifier***
+ **단어 단위**의 input을 받는 **RNN** text classifier
1. [Text_RNN_word_config.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch04_TextRNN_word/Text_RNN_word_config.py):
    - model과 관련한 hyper-parameter 정의
2. [Text_RNN_word_model.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch04_TextRNN_word/Text_RNN_word_model.py):
    - model class
3. [Text_RNN_word_train.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch04_TextRNN_word/Text_RNN_word_train.py):
    - 모형을 학습하고, tensorboard로 summary를 확인 및 학습된 파라미터 저장
4. [Text_RNN_word_predict.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch04_TextRNN_word/Text_RNN_word_train.py):
    - 학습된 모형에 test data를 적용해 성능지표 산출

## Ch05_TextRNN_word_attention
***word level RNN text classifier with attention***
+ **단어 단위**의 input을 받는 RNN text classifier **+ attention**
1. [Text_RNN_word_attention_config.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch05_TextRNN_word_attention/Text_RNN_word_attention_config.py):
    - model과 관련한 hyper-parameter 정의
2. [Text_RNN_word_attention_model.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch05_TextRNN_word_attention/Text_RNN_word_attention_model.py):
    - model class
3. [Text_RNN_word_attention_train.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch05_TextRNN_word_attention/Text_RNN_word_attention_train.py):
    - 모형을 학습하고, tensorboard로 summary를 확인 및 학습된 파라미터 저장
4. [Text_RNN_word_attention_predict.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch05_TextRNN_word_attention/Text_RNN_word_attention_train.py):
    - 학습된 모형에 test data를 적용해 성능지표 산출

## Ch06_TextRCNN_word_attention
***word level RCNN text classifier with attention***
+ **단어 단위**의 input을 받는 **CNN+RNN** text classifier + attention
1. [Text_RCNN_word_attention_config.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch06_TextRCNN_word_attention/Text_RCNN_word_attention_config.py):
    - model과 관련한 hyper-parameter 정의
2. [Text_RCNN_word_attention_model.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch06_TextRCNN_word_attention/Text_RCNN_word_attention_model.py):
    - model class
3. [Text_RCNN_word_attention_train.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch06_TextRCNN_word_attention/Text_RCNN_word_attention_train.py):
    - 모형을 학습하고, tensorboard로 summary를 확인 및 학습된 파라미터 저장
4. [Text_RCNN_word_attention_predict.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch06_TextRCNN_word_attention/Text_RCNN_word_attention_train.py):
    - 학습된 모형에 test data를 적용해 성능지표 산출

