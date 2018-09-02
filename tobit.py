#TOBIY MODEL WITH CENSORING FROM BELOW AND ABOVE AT DIFFERENT POINTS
import numpy as np
import tensorflow as tf
import math as m
from keras.layers import Input, Dense
from keras.models import Model
pi = tf.constant(m.pi)

NUM_SAMPLES = 30000
BATCH_SIZE = 300
NUM_OF_BATCHES = NUM_SAMPLES / BATCH_SIZE

NUMBER_OF_COLUMNS = 2
LAYER_1_NEURONS =  100

#generative model parameteres
B1 = 5
B2 = -6
SIGMA = 5
C = 0.5

#y_train is latent variable
x_train = np.random.randn(NUM_SAMPLES,2)
y_train = x_train[:,0]*B1 + x_train[:,1]*B2 + np.random.randn(NUM_SAMPLES)*SIGMA
y_train = y_train.reshape(NUM_SAMPLES,-1)

#creating hypothethical censor points
censor_train = np.random.randn(NUM_SAMPLES,1)*SIGMA
#determine is it censoring from below or from abovee
down_censor = (censor_train > y_train).astype(int)
up_censor = (censor_train < y_train).astype(int)
#selecting wich observations are actually censored.
censored_ind = np.random.choice(2,(NUM_SAMPLES,1))
down_censor = censored_ind*down_censor
up_censor = censored_ind*up_censor
#creating observable labels
y_train[censored_ind == 1] = censor_train[censored_ind == 1]


### TENSORFLOW  ###
#placeholders
x = tf.placeholder (tf.float32, [None, 2])
y_ = tf.placeholder (tf.float32, [None,1])
up_ind = tf.placeholder( tf.float32, [None,1])
down_ind = tf.placeholder( tf.float32, [None,1])
#censor = tf.placeholder(tf.float32, [None,1])

#variables
#variables for linear model
b = tf.Variable(tf.ones([2,1]))
#in MLE estimation we also estimate variance of our normal distribution
sigma = tf.Variable(1.)

#we can use neural network instead of a simple linear model
layer1_weights = tf.Variable(tf.truncated_normal(
  [2, LAYER_1_NEURONS], stddev=0.1))
layer1_biases = tf.Variable(tf.zeros([LAYER_1_NEURONS]))

output_weights = tf.Variable(tf.truncated_normal(
  [LAYER_1_NEURONS, 1], stddev=0.1))

###------- MODEL------###

#LINEAR
y = tf.matmul(x,b)

#NEURAL NETWORK

#layer1 = tf.matmul(x, layer1_weights) + layer1_biases
#y = tf.matmul(layer1,output_weights)

#W CAN ALSO USE KERAS TO DEFINE NN
#layer1 = Dense(64, activation='linear')(x)
#layer2 = Dense(64, activation='linear')(layer1)
#y = Dense(1, activation= 'linear')(layer2)

###--------------###

###------ MLE ------###
#we use normal distribution when calculating maximum likelihood estimator
normaldist = tf.distributions.Normal(loc=0.,scale=1.)

not_censored_log_argument = normaldist.prob((y_ - y)/sigma)/sigma
up_censored_log_argument = 1 - normaldist.cdf((y_ - y)/sigma)
down_censored_log_argument = normaldist.cdf((y_ - y)/sigma)

not_censored_log_argument = tf.clip_by_value(not_censored_log_argument,0.0000001, 10000000)
up_censored_log_argument = tf.clip_by_value(up_censored_log_argument,0.0000001, 10000000)
down_censored_log_argument = tf.clip_by_value(down_censored_log_argument,0.0000001, 10000000)

loglike = tf.log(not_censored_log_argument)*(1 - up_ind)*(1 - down_ind) + tf.log(up_censored_log_argument)*up_ind*(1 - down_ind) + tf.log(down_censored_log_argument)*down_ind*(1 - up_ind)
loglike2 = tf.reduce_sum(loglike)
loss = -loglike2
###-------------###

train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(500):
        _ , lossy = sess.run([train_step,loss], feed_dict={x: x_train, y_: y_train, up_ind: up_censor, down_ind: down_censor})
        print(step, sess.run(b), sess.run(sigma), lossy)
