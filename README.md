# Text-classification-with-CNN-RNN
Text classification with CNN, RNN, RCNN model by character, word level

-----------------------------------------------------------

Tensorflow를 이용해 한글 text classifier를 CNN, RNN을 이용해 만든 내용입니다.

## Ch01_Data_load
**data link: https://drive.google.com/open?id=1vdvedBSAcVU8Dbjzuow6cL_9Tkc9zGVM**

(전에 공부용으로 수집한 데이터 인데, 문제될 경우 내리겠습니다.) -출처: W 영화 사이트
1. data_preprocessing.py: 텍스트 기본 전처리 진행
    - 특수문자 제거
    - corpus의 문장 길이 10분위수 ~ 95분위수만 사용
    - 2점 이하: 부정, 5점: 긍정으로 나누어 label 비율을 최대한 반반으로 설정
2. data_load.py: 데이터 불러오기
3. Jaso_mapping_utils.py: 텐서에서 자소를 onehot vector로 변환
    - 자소 단위로 input을 받는 모형에 대해 적용
4. make_VocabularyProcessor.py: 텐서에서 단어를 index로 변환
    - 제공하는 VocabularyProcessor를 사용하여 객체 생성
5. utils.py: layers, batch generate, tokenizer 등 사용할 함수 정의
