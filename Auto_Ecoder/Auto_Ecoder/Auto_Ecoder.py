import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("MNIST_data/")
X_train = tf.placeholder(tf.float32,[None,784])
def addLayer(inputData, insize, outsize):
    W = tf.Variable(tf.random_normal([insize,outsize],stddev = 0.1))
    b = tf.Variable(tf.constant(0.1,shape = [outsize]))
    W_add_b = tf.nn.relu(tf.matmul(inputData,W)+b)
    return W_add_b
def encoder(input_data):
    Layer_1 = addLayer(input_data,784,256)
    Layer_2 = addLayer(Layer_1,256,128)
    Layer_3 = addLayer(Layer_2,128,64)
    return Layer_3
def decoder(input_data):
    Layer_1 = addLayer(input_data,64,128)
    Layer_2 = addLayer(Layer_1,128,256)
    Layer_3 = addLayer(Layer_2,256,784)
    return Layer_3
X_encoder = encoder(X_train)
X_decoder = decoder(X_encoder)
loss = tf.reduce_mean(tf.pow(X_train-X_decoder,2))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Begin_Train:")
for i in range(40000):
    batch = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={X_train:batch[0]})
    if i%400 == 0:
        print(i/400)
        print(sess.run(loss,feed_dict={X_train:batch[0]}))
print("进行解码测试")
y_pred = X_decoder
encode_decode = sess.run(y_pred, feed_dict={X_train: mnist.test.images[:20]})
fig,a = plt.subplots(2, 20, figsize=(20, 2))
for i in range(20):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
plt.show()