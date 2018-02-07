from training_base import training_base
from Losses import loss_NLL, bce_weighted_dex, cce_weighted_dex, cce_weighted, moments_weighted
from Losses import metric_means, metric_variances, all_moments_weighted, metric_mo1, metric_mo2, metric_mo3, metric_mo4
from modelTools import fixLayersContaining
import tensorflow as tf

import pdb
import numpy as np
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#gpu_options = tf.GPUOptions(allow_growth=True) #,per_process_gpu_memory_fraction=0.1)
#s = tf.InteractiveSession(config=tf.ConfigProto())#gpu_options=gpu_options))


def combined_ccemv(y_true, y_pred):
    '''
    :param y_true: [isB, isBB, isLeptB, isC, isUDS, isG,  isData, eventweight]
    :param y_pred: [prob_isB, prob_isBB, prob_isLeptB, prob_isC, prob_isUDS, prob_isG]
    :return: combined loss function of cross entropy and moments (means and variances)
    '''

    weight_ce = 1.
    weight_m = 2.
    weight_v = 5.

    y_true_dlw =  y_true[:,-2:]  #domain label and eventweight

    return moments_weighted(y_true_dlw,y_pred,[weight_m,weight_v]) + weight_ce * cce_weighted_dex(y_true, y_pred)

def combined_cce_moments(y_true, y_pred):
    '''
    :param y_true: [isSignal_1, isSignal_2, ... ,  isData, eventweight]
    :param y_pred: [prob_isSignal_1, prob_isSignal_2, ...]
    :return: combined loss function of cross entropy and first four moments
    '''

    weight_ce = 1.
    weight_m = 0.6
    weight_v = 0.6
    weight_s = 0.15
    weight_k = 0.1

    y_true_dlw = y_true[:, -2:]  # domain label and eventweight

    return all_moments_weighted(y_true_dlw, y_pred, [weight_m, weight_v, weight_s, weight_k]) + weight_ce * cce_weighted_dex(y_true, y_pred)


### Main ------------------------------------------------------------------------------------

train=training_base()

train.loadModel("/storage/c/dwalter/data/TFModels/DF_2016Boost/KERAS_model.h5")

model = train.keras_model
#model.summary()

pdb.set_trace()

#freeze all layers but some specific ones

fixLayersContaining(model,"dense",invert=True)

#for layer in model.layers:
#    layer.trainable = False

#model.get_layer("df_dense7").trainable = True
#model.get_layer("df_dense_batchnorm7").trainable = True
#model.get_layer("ID_pred").trainable = True

#model.summary()

#pdb.set_trace()

#train.compileModel(learningrate=0.0001, loss=[combined_ccemv, loss_NLL], metrics={'ID_pred':[cce_weighted_dex, metric_means, metric_variances]},loss_weights=[1., 0.0])
#train.compileModel(learningrate=0.0001, loss=[combined_cce_moments, loss_NLL], metrics={'ID_pred':[cce_weighted_dex, metric_mo1, metric_mo2, metric_mo3, metric_mo4]},loss_weights=[1., 0.0])
train.compileModel(learningrate=0.0001, loss=[combined_ccemv, loss_NLL], metrics={'ID_pred':[cce_weighted_dex, metric_mo1, metric_mo2, metric_mo3, metric_mo4]},loss_weights=[1., 0.0])


#gather the training samples
sample_inputs = train.train_data.getAllFeatures()
sample_labels = train.train_data.getAllLabels()
sample_weights = train.train_data.getAllWeights() #[weight, eventweight, domainLabel]


#shuffle data one time before fit, because validation data has to be shuffled too, this takes some time ...
print('shuffle data ...')
from Helpers import shuffle_in_unison
sample_inputs[0], sample_inputs[1], sample_inputs[2], sample_inputs[3], sample_inputs[4], sample_labels[0], sample_labels[1], sample_weights[0], sample_weights[1], sample_weights[2] = shuffle_in_unison((sample_inputs[0], sample_inputs[1], sample_inputs[2], sample_inputs[3], sample_inputs[4], sample_labels[0], sample_labels[1], sample_weights[0], sample_weights[1], sample_weights[2]))
print('done shuffling data')

#mc_inputs = [sample_inputs[0][sample_weights[2]==0], sample_inputs[1][sample_weights[2]==0], sample_inputs[2][sample_weights[2]==0],sample_inputs[3][sample_weights[2]==0],sample_inputs[4][sample_weights[2]==0]]
#mc_labels = [sample_labels[0][sample_weights[2]==0], sample_labels[1][sample_weights[2]==0]]
#mc_eventweights = np.array([sample_weights[1][sample_weights[2]==0]]).T

#mc_labelsAndWeight = np.concatenate((mc_labels[0], mc_eventweights), axis=1)  # attach eventweight to the ID labels



#flavour labels, domain labels, and eventweights
sample_fldlw = np.concatenate((sample_labels[0], np.array([sample_weights[2]]).T, np.array([sample_weights[1]]).T), axis=1)
sample_dlw = np.concatenate((np.array([sample_weights[2]]).T, np.array([sample_weights[1]]).T), axis=1)

print("compile model")


history = model.fit(x=sample_inputs, y=[sample_fldlw,sample_labels[1]], batch_size=100000, epochs=150, validation_split=0.1)

#recompile without metrics, they would cause trouble when loading the model, (can't load a list of metrics)
train.compileModel(learningrate=0.0001, loss=[combined_ccemv, loss_NLL],loss_weights=[1., 0.0])
#train.compileModel(learningrate=0.0001, loss=[combined_cce_moments, loss_NLL],loss_weights=[1., 0.0])


model.save(train.outputDir+"KERAS_model.h5")
pickle.dump(history.history,open(train.outputDir+"Keras_history.p","wb"))


print("completed training")