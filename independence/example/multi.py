'''
file for testing custom loss functions and layers on random samples
'''
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

import pdb
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras import backend as K
from keras.engine import Layer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

print ('starting multi')


# loss functions
def loss_moment(y_in, x):
    ''' loss function for categorical decision problem '''

    # h is the histogram vector "one hot encoded" (5 bins in this case), technically part of the "truth" y
    h = y_in[:, 1:]
    y = y_in[:, :1]
    isData = y_in[:, 1:2]
    # The below counts the entries in the histogram vector
    h_entr = K.sum(h, axis=0)

    ## first moment ##

    # Multiply the histogram vectors with estimated probability x
    h_fill = h * x

    # Sum each histogram vector
    Sum = K.sum(h_fill, axis=0)

    # Devide sum by entries (i.e. mean, first moment)
    Sum = Sum / h_entr

    # Devide per vector mean by average mean
    Sum = Sum / K.mean(x)

    ## second moment

    x2 = x - K.mean(x)
    h_fill2 = h * x2 * x2
    Sum2 = K.sum(h_fill2, axis=0)
    Sum2 = Sum2 / h_entr
    Sum2 = Sum2 / K.mean(x2 * x2)

    ## third moment

    x3 = x - K.mean(x)
    h_fill3 = h * x3 * x3 * x3
    Sum3 = K.sum(h_fill3, axis=0)
    Sum3 = Sum3 / h_entr
    Sum3 = Sum3 / K.mean(x2 * x2 * x2)

    ## fourth moment

    x4 = x - K.mean(x)
    h_fill4 = h * x4 * x4 * x4 * x4
    Sum4 = K.sum(h_fill4, axis=0)
    Sum4 = Sum4 / h_entr
    Sum4 = Sum4 / K.mean(x4 * x4 * x4 * x4)

    # Xentro = K.mean(K.binary_crossentropy(x,y)*y_in[:,2:3], axis=-1)
    return K.mean(K.square(Sum - 1)) + K.mean(K.square(Sum2 - 1)) + K.mean(K.square(Sum3 - 1)) + K.mean(
        K.square(Sum4 - 1))


def mse(y_true, y_pred):
    '''mean square error which only takes mc into account'''
    isSignal = y_true[:, 0]
    isSignalPred = y_pred[:, 0]
    isData = y_true[:, -2]
    weight = y_true[:, -1]
    return K.sum(weight * K.square(isSignalPred - isSignal) * (1 - isData)) / K.sum((1 - isData) * weight)


def bc(y_true, y_pred):
    '''binary crossentropy which only takes mc into account'''
    isSignal = y_true[:, 0]
    isSignalPred = y_pred[:, 0]
    isData = y_true[:, -2]
    weight = y_true[:, -1]
    return K.sum(weight * K.binary_crossentropy(isSignalPred, isSignal) * (1 - isData)) / K.sum((1 - isData) * weight)


def means_distance(y_true, y_pred):
    '''punishment linear to the difference of the means of the output distributions from data and mc'''
    isSignalPred = y_pred[:, 0]
    isData = y_true[:, -2]
    weight = y_true[:, -1]
    mean_mc = K.sum((1 - isData) * weight * isSignalPred) / K.sum((1 - isData) * weight)
    mean_data = K.sum(isData * weight * isSignalPred) / K.sum(isData * weight)
    return abs(mean_mc - mean_data)


def variance_distance(y_true, y_pred):
    '''punishment linear to the difference of the variances of the output distributions from data and mc'''
    isSignalPred = y_pred[:, 0]
    isData = y_true[:, -2]
    weight = y_true[:, -1]
    mean_mc = K.sum((1 - isData) * weight * isSignalPred) / K.sum((1 - isData) * weight)
    mean_data = K.sum(isData * weight * isSignalPred) / K.sum(isData * weight)
    var_mc = K.sum(weight * ((isSignalPred - mean_mc) * (1 - isData)) ** 2) / (K.sum((1 - isData) * weight))
    var_data = K.sum(weight * ((isSignalPred - mean_data) * isData) ** 2) / (K.sum(isData * weight))
    return abs(var_mc - var_data)


# metrics
def c_accuracy(y_true, y_pred):
    isSignal = y_true[:, 0]
    isSignalPred = y_pred[:, 0]
    return K.mean(K.equal(isSignal, K.round(isSignalPred)), axis=-1)


def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)
    return y


class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''

    def __init__(self, hp_lambda=1., **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# generating dummy mc/data

def sample(data, flip, know):
    n = 400000
    a_1 = np.random.randn(n / 2 + int(flip) * n / 4) + 1.5  # process 1 (lets say this is background)
    a_2 = np.random.randn(n / 2 - int(flip) * n / 4) + 0  # process 2 (lets say this is signal)
    b_1 = np.random.randn(n / 2 + int(flip) * n / 4) + 1.5
    b_2 = np.random.randn(n / 2 - int(flip) * n / 4)
    w_1 = np.ones(n / 2 + int(flip) * n / 4)  # weights for signal
    w_2 = np.ones(n / 2 - int(flip) * n / 4)
    if data == 0:  # weights for mc to represent the data 1 composition
        w_1 *= 1. / 3
        w_2 *= 3.
    if data:  # data has in parts of the sample a shift in sigma or mu
        a_1 = np.random.randn(n / 2 + int(flip) * n / 4) + 2.5
        a_2 = np.random.randn(n / 2 - int(flip) * n / 4) + 1.5
        b_1 = np.random.randn(n / 2 + int(flip) * n / 4) * 2 + 1.5
    y_1 = np.zeros(n / 2 + int(flip) * n / 4)
    y_2 = np.ones(n / 2 - int(flip) * n / 4)
    y = np.concatenate((y_1, y_2), axis=0)  # output
    a = np.concatenate((a_1, a_2), axis=0)  # feature 1
    b = np.concatenate((b_1, b_2), axis=0)  # feature 2
    w = np.concatenate((w_1, w_2), axis=0)
    d = np.zeros(n)  # d indicates if sample is data
    if data: d = np.ones(n)
    f = np.ones(n)  # f indicates if flipped
    if flip == 1: f = np.zeros(n)
    k = d  # k indicates if one knows if sample is mc
    if know == 0: k = np.zeros(n)
    all = np.vstack((a, b, k, y, f, np.ones(n), d, w)).T
    return all


def samples(flip, know=1):
    mc1 = sample(0, 1, know)  # 50,000 signal and 150,000 bg
    mc2 = sample(0, -1, know)  # 150,000 signal and  50,000 bg
    data1 = sample(1, -1, know)  # 150,000 signal and  50,000 bg
    data2 = sample(1, 1, know)  # 50,000 signal and 150,000 bg
    datamc = np.vstack((mc1, data1))
    np.random.shuffle(datamc)
    mix_X = datamc[:, :3]
    mix_Y = datamc[:, 3:]
    return mix_X, mix_Y, mc1[:, :3], mc1[:, 3:], data1[:, :3], data1[:, 3:], mc2[:, :3], mc2[:, 3:], data2[:,
                                                                                                     :3], data2[:, 3:]


### some plotting tools
def plotSamples():
    '''gives some information of the input sample'''

    mc = np.concatenate((mc1_X, mc2_X))
    data = np.concatenate((data1_X, data2_X))
    bins = np.arange(-10, 10, 0.4)

    plt.hist(mc1_X[:, 0], bins, label='mc1', histtype='step', color='#d62728')
    plt.hist(data1_X[:, 0], bins, label='data1', histtype='step', color='#1f77b4')
    plt.xlabel('feature 1')
    plt.ylabel('events')
    plt.legend()
    # plt.show()
    plt.savefig('feature1.png')
    plt.cla()

    n, x, _ = plt.hist(mc1_X[:, 1], bins, label='mc1', histtype='step', color='#d62728')
    n2, x2, _2 = plt.hist(data1_X[:, 1], bins, label='data1', histtype='step', color='#1f77b4')
    plt.xlabel('feature 2')
    plt.ylabel('events')
    plt.legend()
    # plt.show()
    plt.savefig('feature2.png')
    plt.cla()

    plt.hist2d(my_X[:, 0], my_X[:, 1], bins=40, label='mc and data')
    plt.title('mixed sample input ')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    # plt.show()
    # plt.savefig('features.png')
    plt.cla()


def plotHistory(key):
    plt.plot(history.history[key], label='training')
    plt.plot(history.history['val_' + key], label='validation')
    plt.xlabel('epoch')
    plt.ylabel(key)
    plt.legend()
    # plt.show()
    plt.savefig(key + '.png')
    plt.cla()


def plotOutput():
    bins = np.arange(0, 1.05, 0.05)
    plt.hist(output_data1, bins, label='data 1', histtype='step', color='#1f77b4')
    plt.hist(output_data2, bins, label='data 2', histtype='step', color='#2ca02c')
    plt.hist(output_mc1, bins, label='mc 1', histtype='step', color='#d62728')
    plt.hist(output_mc2, bins, label='mc 2', histtype='step', color='#ff7f0e')
    plt.text(0.2, plt.ylim()[1] * 0.9, 'acc data1 = ' + str(score_data1[1]))
    plt.text(0.2, plt.ylim()[1] * 0.85, 'acc data2 = ' + str(score_data2[1]))
    plt.text(0.2, plt.ylim()[1] * 0.8, 'acc mc1 = ' + str(score_mc1[1]))
    plt.text(0.2, plt.ylim()[1] * 0.75, 'acc mc2 = ' + str(score_mc2[1]))
    plt.legend()
    plt.title(title)
    # plt.show()
    plt.savefig('output_all.png')
    plt.cla()

    output_data = np.concatenate((output_data1, output_data2))
    output_mc = np.concatenate((output_mc1, output_mc2))

    plt.hist(output_data1, bins, label='data1 output', histtype='step', color='blue')
    plt.hist(output_mc1, bins, label='mc1 output', histtype='step', color='red')
    plt.legend()
    # plt.show()
    plt.title(title)
    plt.savefig('output.png')
    plt.cla()


# models
def adversarial1():
    # from data_bad import GradientReversal, reverse_gradient

    print ('build model')

    Inputs = Input((3,))
    X = Dense(10, activation='relu', input_shape=(3,))(Inputs)
    X = Dense(20, activation='relu')(X)
    Xa = Dense(20, activation='relu')(X)
    X = Dense(10, activation='relu')(Xa)
    X = Dense(1, activation='sigmoid')(X)
    Ad = GradientReversal()(Xa)
    Ad = Dense(10, activation='relu')(Ad)
    Ad = Dense(10, activation='relu')(Ad)
    Ad = Dense(10, activation='relu')(Ad)
    Ad = Dense(1, activation='sigmoid')(Ad)
    Ad1 = GradientReversal()(Xa)
    Ad1 = Dense(10, activation='relu')(Ad1)
    Ad1 = Dense(10, activation='relu')(Ad1)
    Ad1 = Dense(10, activation='relu')(Ad1)
    Ad1 = Dense(1, activation='sigmoid')(Ad1)

    predictions = [X, Ad, Ad1]
    model = Model(inputs=Inputs, outputs=predictions)

    print ('compile model')

    model.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], optimizer='adam',
                  loss_weights=[1., 50., 50.])

    return model


def feedforward():
    '''Inputs:  2, one for each feature
       Outputs: 2, one for signal predictions, one for domain predictions (is Data or not)
       there are three outputs to control three competetive loss functions'''

    print ('built model')

    Inputs = Input((2,))
    X = Dense(10, activation='relu', input_shape=(2,))(Inputs)
    X = Dense(20, activation='relu')(X)
    X = Dense(1, activation='sigmoid')(X)

    def combined_cemv(y_true, y_pred):
        weight_ce = 1.
        weight_m = 0.5
        weight_v = 0.
        return weight_ce * bc(y_true, y_pred) + weight_m * means_distance(y_true,
                                                                          y_pred) + weight_v * variance_distance(y_true,
                                                                                                                 y_pred)

    model = Model(inputs=Inputs, outputs=X)
    model.compile(loss=combined_cemv,
                  optimizer='adam',
                  metrics=[c_accuracy, means_distance, variance_distance, bc],
                  )
    return model


from keras.models import load_model

global_loss_list = {}
# global_loss_list['loss_moment']=loss_moment
global_loss_list['GradientReversal'] = GradientReversal()
# model = load_model('my_model_0.h5',custom_objects=global_loss_list)

print ('make samples')
my_X, my_Y, mc1_X, mc1_Y, data1_X, data1_Y, mc2_X, mc2_Y, data2_X, data2_Y = samples(-1, know=1)
# pdb.set_trace()
print ('got samples')
print (my_Y[:, 1:2])
print ('got samples')
print (my_Y[:, 2:3] - my_Y[:, 1:2])

weigh_crap = (my_Y[:, 2:3] - my_Y[:, 1:2]).ravel()  # 1-flip

# model = adversarial1()
# model.fit(my_X.astype('float32'),
#          [my_Y[:,:1].astype('float32'), my_Y[:,3:4].ravel(), my_Y[:,3:4].ravel()],
#          batch_size=5000,
#          epochs=50,
#          verbose=1,
#          validation_split=0.3,
#          sample_weight=[my_Y[:,3:4].ravel(),my_Y[:,1:2].ravel() ,weigh_crap])


model = feedforward()
y = np.vstack((my_Y[:, 0], my_Y[:, 3], my_Y[:, 4])).T

# y = np.concatenate((my_Y[:,0],my_Y[:,3]),axis=1)
history = model.fit(my_X[:, :2].astype('float32'),
                    y,
                    batch_size=50000,
                    epochs=50,
                    verbose=1,
                    validation_split=0.3)

# pdb.set_trace()
title = 'CEM'
print("make directory and save plots")
directory = os.path.dirname('./plots_' + title + '/')
# make a canvas, draw, and save it
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

plotSamples()

plotHistory('c_accuracy')
plotHistory('loss')
plotHistory('bc')
plotHistory('means_distance')
plotHistory('variance_distance')

print('predict')
output_data1 = model.predict(data1_X[:, :2])
output_mc1 = model.predict(mc1_X[:, :2])
output_data2 = model.predict(data2_X[:, :2])
output_mc2 = model.predict(mc2_X[:, :2])

data1_y = np.vstack((data1_Y[:, 0], data1_Y[:, 3], data1_Y[:, 4])).T
mc1_y = np.vstack((mc1_Y[:, 0], mc1_Y[:, 3], mc1_Y[:, 4])).T
data2_y = np.vstack((data2_Y[:, 0], data2_Y[:, 3], data2_Y[:, 4])).T
mc2_y = np.vstack((mc2_Y[:, 0], mc2_Y[:, 3], mc2_Y[:, 4])).T

score_data1 = model.evaluate(data1_X[:, :2], data1_y)
score_mc1 = model.evaluate(mc1_X[:, :2], mc1_y)
score_data2 = model.evaluate(data2_X[:, :2], data2_y)
score_mc2 = model.evaluate(mc2_X[:, :2], mc2_y)

print('scores: data1 = ', score_data1[1], 'data2 = ', score_data2[1], 'mc1 = ', score_mc1[1], 'mc2 = ', score_mc2[1])

plotOutput()

# np.save('outfile_mc_data.npy', ouput[0])
# np.save('outfile_data_data.npy', ouput_data[0])
# plt.imshow(heatmap, cmap='hot', interpolation='nearest',norm=matplotlib.colors.LogNorm(),extent =extent )
# plt.draw()
# plt.subplot(122)

import sklearn
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(mc1_Y[:, 0], output_mc1)
auc_score_mc = 1. - auc(fpr, tpr)
plt.semilogy(tpr, fpr, label='mc', color='#d62728')
fprd, tprd, thresholds2 = roc_curve(data1_Y[:, 0], output_data1)
auc_score_data = 1. - auc(fprd, tprd)
plt.semilogy(tprd, fprd, label='data 1', color='#1f77b4')
fprd2, tprd2, thresholds3 = roc_curve(data2_Y[:, 0], output_data2)
auc_score_data2 = 1. - auc(fprd2, tprd2)
plt.semilogy(tprd2, fprd2, label='data 2', color='#2ca02c')  # semilogy
print('area under roc curve in mc: ', auc_score_mc, ' in data 1: ', auc_score_data, 'in data 2: ', auc_score_data2)
plt.legend(loc=4)
plt.xlabel('true positive rate')
plt.ylabel('false positive rate')
plt.title(title)
plt.text(0.2, plt.ylim()[1] * 0.5, 'auc mc = ' + str(auc_score_mc))
plt.text(0.2, plt.ylim()[1] * 0.25, 'auc data 1 = ' + str(auc_score_data))
plt.text(0.2, plt.ylim()[1] * 0.125, 'auc data 2 = ' + str(auc_score_data2))

# plt.show()
plt.savefig('roc.png')

pdb.set_trace()

ax = plt.subplot(122)

tprd_0 = np.load('outfile_tprd.npy')
fprd_0 = np.load('outfile_fprd.npy')
ax.plot(fprd_0, tprd_0, label='data_0')
tpr_0 = np.load('outfile_tpr.npy')
fpr_0 = np.load('outfile_fpr.npy')
ax.plot(fpr_0, tpr_0, label='mc_0')
ax.plot(fprd, tprd, linestyle='--', label='data')
ax.plot(fpr, tpr, linestyle='--', label='mc''r--')
ax.legend()

np.save('outfile_fpr_data_DA.npy', fpr)
np.save('outfile_tpr_data_DA.npy', tpr)
np.save('outfile_fprd_data_DA.npy', fprd)
np.save('outfile_tprd_data_DA.npy', tprd)

tprd_d = np.load('outfile_tprd_data.npy')
fprd_d = np.load('outfile_fprd_data.npy')
tpr_d = np.load('outfile_tpr_data.npy')
fpr_d = np.load('outfile_fpr_data.npy')

ax.plot(fprd_d, tprd_d, color='r', linestyle=':', label='data (data)')
ax.plot(fpr_d, tpr_d, color='b', linestyle=':', label='mc (data)')
plt.title('ROC', size=20)
ax.legend(prop={'size': 21})
axes = plt.gca()
axes.set_xlim([0., 0.4])
axes.set_ylim([0., 1.])
plt.draw()

import time

plt.pause(0.001)
input("Press [enter] to continue.")

plt.figure(2)
heatmap2, xedges, yedges = np.histogram2d(mc_X[:, 0:1].ravel(), mc_X[:, 1:2].ravel(), bins=200)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
print(extent)
extent = [-5, 10, -5, 5]
plt.title('mc', size=20)
plt.imshow(heatmap2, cmap='hot', interpolation='nearest', norm=matplotlib.colors.LogNorm(), extent=extent)
# plt.savefig('mc.pdf')

# plt.figure(3)
# plt.title('data', size=20)
# heatmap2, xedges, yedges = np.histogram2d(data_X[:,0:1].ravel(),data_X[:,1:2].ravel(), bins=200)
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# plt.imshow(heatmap2, cmap='hot', interpolation='nearest',norm=matplotlib.colors.LogNorm(),extent =extent)
# plt.savefig('data.pdf')


plt.figure(3)
plt.hist([ouput[0], ouput_data[0]], label=['mc', 'data'], bins=20)
plt.legend(prop={'size': 21})
plt.title('probability')
plt.draw()
plt.savefig('prob.pdf')
import time

plt.pause(0.001)
input("Press [enter] to continue.")

model.save('my_model_0.h5')  # creates a HDF5 file 'my_model.h5'
del model

# model.fit(my_X.astype('float32'),[my_Y[:,:1].astype('float32'), my_X[:,2:].astype('float32'), my_X[:,2:].astype('float32')], batch_size=10000, epochs=20, verbose=1,validation_split=0.3,sample_weight=[my_Y[:,1:2].ravel(),my_Y[:,1:2].ravel(), (my_Y[:,1:2]-my_Y[:,2:3])*(my_Y[:,1:2]-my_Y[:,2:3]).ravel() ])



















