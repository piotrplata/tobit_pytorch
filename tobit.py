import numpy as np
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

NUM_SAMPLES = 30000

# generative model parameteres
B = [5, -6, 3]
SIGMA = 5

input_size = len(B)
# y_train is latent variable
x_train = np.random.randn(NUM_SAMPLES, input_size)
y_train = np.sum(x_train * B, axis=1) + np.random.randn(NUM_SAMPLES) * SIGMA

y_train = y_train.reshape(NUM_SAMPLES, -1)

# creating hypothethical censor points
censor_train = np.random.randn(NUM_SAMPLES, 1) * SIGMA
# determine is it censoring from below or from abovee
down_censor = (censor_train > y_train).astype(int)
up_censor = (censor_train < y_train).astype(int)
# selecting wich observations are actually censored.
censored_ind = np.random.choice(2, (NUM_SAMPLES, 1))
down_censor = censored_ind * down_censor
up_censor = censored_ind * up_censor
# creating observable labels
y_train[censored_ind == 1] = censor_train[censored_ind == 1]


class Model(object):
    def __init__(self, input_size):
        # variables for linear model
        self.b = tf.Variable(tf.ones([input_size, 1]))
        # in MLE estimation we also estimate variance of our normal distribution
        self.sigma = tf.Variable(1.)

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        return tf.matmul(x, self.b), self.sigma


model = Model(input_size)


def loglike(y, sigma, y_, up_ind, down_ind):
    y = tf.cast(y, tf.float32)
    y_ = tf.cast(y_, tf.float32)
    up_ind = tf.cast(up_ind, tf.float32)
    down_ind = tf.cast(down_ind, tf.float32)

    normaldist = tfp.distributions.Normal(loc=0., scale=1.)

    not_censored_log_argument = normaldist.prob((y_ - y) / sigma) / sigma
    up_censored_log_argument = 1 - normaldist.cdf((y_ - y) / sigma)
    down_censored_log_argument = normaldist.cdf((y_ - y) / sigma)

    not_censored_log_argument = tf.clip_by_value(not_censored_log_argument, 0.0000001, 10000000)
    up_censored_log_argument = tf.clip_by_value(up_censored_log_argument, 0.0000001, 10000000)
    down_censored_log_argument = tf.clip_by_value(down_censored_log_argument, 0.0000001, 10000000)

    loglike = tf.math.log(not_censored_log_argument) * (1 - up_ind) * (1 - down_ind) + tf.math.log(
        up_censored_log_argument) * up_ind * (1 - down_ind) + tf.math.log(down_censored_log_argument) * down_ind * (
                      1 - up_ind)
    loglike2 = tf.reduce_sum(loglike)
    loss = -loglike2
    return loss


optimizer = Adam(0.1)
train_loss = tf.keras.metrics.Mean(name='train_loss')


@tf.function
def train_step(x, y_, up_ind, down_ind):
    with tf.GradientTape() as tape:
        y, sigma = model(x)
        loss = loglike(y, sigma, y_, up_ind, down_ind)
        trainable_variables = [model.b, model.sigma]

        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        train_loss(loss)


EPOCHS = 1000

for epoch in range(EPOCHS):
    train_loss.reset_states()

    train_step(x_train, y_train, up_censor, down_censor)

    template = 'Epoch: {}, Loss: {}, Est_B: {}, Est_SIGMA: {}'

    print(template.format(epoch+1, train_loss.result(), model.b.value(), model.sigma.value()))
