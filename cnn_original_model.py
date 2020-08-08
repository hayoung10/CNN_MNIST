
# coding: utf-8

# # 기계학습 과제2
# ### Tensorflow를 활용한 CNN 구현 및 실험

# #### 필요한 패키지 로드
# #### MNIST 데이터 셋을 읽어오기

# In[2]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False)


# #### CNN 모델 설정

# In[3]:


X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
Y_Label = tf.placeholder(tf.float32, shape=[None, 10])


# #### 첫 번째 Convolution Layer

# In[4]:


Kernel1 = tf.Variable(tf.truncated_normal(shape=[4, 4, 1, 4], stddev=0.1))
Bias1 = tf.Variable(tf.truncated_normal(shape=[4], stddev=0.1))
Conv1 = tf.nn.conv2d(X, Kernel1, strides=[1, 1, 1, 1], padding='SAME') + Bias1
Activation1 = tf.nn.relu(Conv1)
Pool1 = tf.nn.max_pool(Activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# #### 두 번째 Convolution Layer

# In[5]:


Kernel2 = tf.Variable(tf.truncated_normal(shape=[4, 4, 4, 8], stddev=0.1))
Bias2 = tf.Variable(tf.truncated_normal(shape=[8], stddev=0.1))
Conv2 = tf.nn.conv2d(Pool1, Kernel2, strides=[1, 1, 1, 1], padding='SAME') + Bias2
Activation2 = tf.nn.relu(Conv2)
Pool2 = tf.nn.max_pool(Activation2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# #### Fully Connected Layer

# In[6]:


W1 = tf.Variable(tf.truncated_normal(shape=[8 * 7 * 7, 10]))
B1 = tf.Variable(tf.truncated_normal(shape=[10]))
Pool2_flat = tf.reshape(Pool2, [-1, 8 * 7 * 7])
OutputLayer = tf.matmul(Pool2_flat, W1) + B1


# In[1]:





with tf.name_scope("loss"):
    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=OutputLayer))
    train_step = tf.train.AdamOptimizer(0.005).minimize(Loss)
    tf.summary.scalar('loss', Loss)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(OutputLayer, 1), tf.argmax(Y_Label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    print("Start....")
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)  # for 0.8
    merged = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    summary, acc = sess.run([merged, accuracy], feed_dict={X: mnist.test.images, Y_Label: mnist.test.labels})
    for i in range(150):
        trainingData, Y = mnist.train.next_batch(64)
        sess.run(train_step, feed_dict={X: trainingData, Y_Label: Y})
        if i % 10 == 0:
            print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_Label: mnist.test.labels}))
        summary, acc = sess.run([merged, accuracy], feed_dict={X: mnist.test.images, Y_Label: mnist.test.labels})

