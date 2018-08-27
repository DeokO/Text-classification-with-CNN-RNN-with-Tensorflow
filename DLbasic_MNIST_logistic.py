# https://github.com/DeokO/CS20SI-Tensorflow-for-Deep-Learning-Research-Code-Exercise/blob/master/notes03_Regression/ex03_2_logisticRegression.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#Step 1: read in data
#training(55,000), test(10,000), validation(5,000)
MNIST = input_data.read_data_sets('./Ch01_Data_load/data/mnist', one_hot=True)

#Step 2: Define parameters for the model
learning_rate = 0.001
batch_size = 128
n_epochs = 50

#Step 3: create placeholders for features and labels
#input : 28 * 28 = 784, 1 * 784로 제공된다.
#10개의 class가 one hot vector로 제공된다. (0 ~ 9)
#data가 많기 때문에 batch 단위로 update를 진행한다.
X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='label')

#Step 4: create weights and bias
#w는 정규분포 N(0, 0.01**2)로부터 초기화
#b는 0으로 초기화 (나중에 Relu activation을 이용할 경우에는 초기화를 0보다 아주 조금 큰 양수로 초기화 하는 것을 추천함. by Deeplearning book)
#w의 차원은 X, Y의 차원과 관련있다. (Y = tf.matmul(X, w))
#b의 차원은 Y와 관련있다.
w = tf.Variable(initial_value=tf.random_normal(shape=[784, 10], stddev=0.01), name='weights') #784차원을 10차원으로 보내는 것으로 이해하자.
b = tf.Variable(initial_value=tf.zeros(shape=[1, 10]), name='bias') #1차원을 10차원으로

#Step 5: predict Y from X and w, b
#출력값은 softmax를 통해 Y가 어떤 label이 될 확률 분포를 얻게 됨
logits = tf.matmul(X, w) + b

#Step 6: define loss function
#위에서 정의한 logits과 실제 label값인 Y에 softmax cross entropy를 적용해서 entropy를 만들고, 이를 평균내서 loss를 정의한다.
#중요한 것은 logit에 softmax를 실행하지 않은 채로 넣어준다는 것이다. softmax를 계산하고, cross entropy를 계산하는 듯 하다.
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)
tf.summary.scalar('loss', loss)
merge = tf.summary.merge_all()
predict = tf.nn.softmax(logits)

#Step 7: define training op
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("./output/MNIST_logistic/", sess.graph)
    n_batches = int(MNIST.train.num_examples / batch_size) #실제로 1번 epoch에서 돌아갈 배치 횟수

    count = 0
    for i in range(n_epochs): #50회 epoch
        for _ in range(n_batches): #1 epoch에서는 429번의 batch가 계산된다.
            count += 1
            X_batch, Y_batch = MNIST.train.next_batch(batch_size) #batch를 임의로 선택해주는 함수가 내장되어 있음
            # np.shape(X_batch), np.shape(Y_batch)
            _, ll, merged = sess.run([optimizer, loss, merge], feed_dict={X: X_batch, Y:Y_batch})
            writer.add_summary(merged, count)
        print('Epoch{} _ loss : {:.4f}'.format(i, ll)) #predict를 print 하는 부분을 생략하는게 속도를 위해서는 좋을 수도 있겠다.

    #test the model
    n_batches = int(MNIST.test.num_examples / batch_size) #test data에서는 78번의 batch를 계산해야 1번의 epoch가 된다.
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X:X_batch, Y:Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1)) #각각 행단위로 argmax를 구함
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy) #맞는 개수를 더해줌

    print("Accuracy: {:.2%}".format(total_correct_preds / MNIST.test.num_examples)) #전체 개수로 나눠서 Accuracy(정분류율) 계산

    writer.close()

# 연습용이므로 파라미터 saver를 따로 정의하지 않음