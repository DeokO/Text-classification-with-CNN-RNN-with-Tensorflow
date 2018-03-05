# Text-classification-with-CNN-RNN
Text classification with CNN, RNN, RCNN model by character, word level

-----------------------------------------------------------

Tensorflow를 이용해 한글 text classifier를 CNN, RNN을 이용해 만든 내용입니다.

## Ch01_Data_load
### data link: https://drive.google.com/open?id=1vdvedBSAcVU8Dbjzuow6cL_9Tkc9zGVM

(전에 공부용으로 수집한 데이터 인데, 문제될 경우 내리겠습니다.) -출처: W 영화 사이트
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
Model: 자소 단위의 input을 받는 ***CNN text classifier***
1. [Text_CNN_config.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch02_TextCNN/Text_CNN_config.py):
    - model과 관련한 hyper-parameter 정의
2. [Text_CNN_model.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch02_TextCNN/Text_CNN_model.py):
    - model class
3. [Text_CNN_train.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch02_TextCNN/Text_CNN_train.py):
    - 모형을 학습하고, tensorboard로 summary를 확인 및 학습된 파라미터 저장
4. [Text_CNN_predict.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch02_TextCNN/Text_CNN_train.py):
    - 학습된 모형에 test data를 적용해 성능지표 산출

## Ch03_TextRNN
Model: 자소 단위의 input을 받는 ***RNN text classifier***
1. [Text_RNN_config.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch03_TextRNN/Text_RNN_config.py):
    - model과 관련한 hyper-parameter 정의
2. [Text_RNN_model.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch03_TextRNN/Text_RNN_model.py):
    - model class
3. [Text_RNN_train.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch03_TextRNN/Text_RNN_train.py):
    - 모형을 학습하고, tensorboard로 summary를 확인 및 학습된 파라미터 저장
4. [Text_RNN_predict.py](https://github.com/DeokO/Text-classification-with-CNN-RNN-with-Tensorflow/blob/master/Ch03_TextRNN/Text_RNN_train.py):
    - 학습된 모형에 test data를 적용해 성능지표 산출
