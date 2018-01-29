from training_base import training_base
from models import convolutional_model_broad_map
from Losses import loss_NLL, bce_weighted_dex, cce_weighted_dex, cce_weighted, moments_weighted
from DeepJet_callbacks import DeepJet_callbacks
from modelTools import fixLayersContaining

import pdb
import numpy as np
import tensorflow as tf
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#gpu_options = tf.GPUOptions(allow_growth=True) #,per_process_gpu_memory_fraction=0.1)
#s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


def combined_ccemv(y_true, y_pred):
    '''
    :param y_true: [isB, isBB, isLeptB, isC, isUDS, isG,  isData, eventweight]
    :param y_pred: [prob_isB, prob_isBB, prob_isLeptB, prob_isC, prob_isUDS, prob_isG]
    :return: combined loss function of cross entropy and moments (means and variances)
    '''

    weight_ce = 1.
    weight_m = 0.1
    weight_v = 0.

    y_true_dlw =  y_true[:,-2:]  #domain label and eventweight

    return moments_weighted(y_true_dlw,y_pred,[weight_m,weight_v]) + weight_ce * cce_weighted_dex(y_true, y_pred)

def metric_means(y_true, y_pred):
    weight_m = 1.
    weight_v = 0.

    y_true_dlw =  y_true[:,-2:]  #domain label and eventweight

    return moments_weighted(y_true_dlw,y_pred,[weight_m,weight_v])

def metric_variances(y_true, y_pred):
    weight_m = 0.
    weight_v = 1.

    y_true_dlw =  y_true[:,-2:]  #domain label and eventweight

    return moments_weighted(y_true_dlw,y_pred,[weight_m,weight_v])

### Main ------------------------------------------------------------------------------------

train=training_base()

train.loadModel("/storage/c/dwalter/data/TFModels/DF_2016Boost/KERAS_model.h5")

model = train.keras_model
#model.summary()

#freeze all layers but some specific ones

fixLayersContaining(model,"dense",invert=True)

#for layer in model.layers:
#    layer.trainable = False

#model.get_layer("df_dense7").trainable = True
#model.get_layer("df_dense_batchnorm7").trainable = True
#model.get_layer("ID_pred").trainable = True

#model.summary()

#gather the training samples
sample_inputs = train.train_data.getAllFeatures()
sample_labels = train.train_data.getAllLabels()
sample_weights = train.train_data.getAllWeights() #[weight, eventweight, domainLabel]

#mc_inputs = [sample_inputs[0][sample_weights[2]==0], sample_inputs[1][sample_weights[2]==0], sample_inputs[2][sample_weights[2]==0],sample_inputs[3][sample_weights[2]==0],sample_inputs[4][sample_weights[2]==0]]
#mc_labels = [sample_labels[0][sample_weights[2]==0], sample_labels[1][sample_weights[2]==0]]
#mc_eventweights = np.array([sample_weights[1][sample_weights[2]==0]]).T

#mc_labelsAndWeight = np.concatenate((mc_labels[0], mc_eventweights), axis=1)  # attach eventweight to the ID labels


print("compile model")

#pdb.set_trace()


#flavour labels, domain labels, and eventweights
sample_fldlw = np.concatenate((sample_labels[0], np.array([sample_weights[2]]).T, np.array([sample_weights[1]]).T), axis=1)
sample_dlw = np.concatenate((np.array([sample_weights[2]]).T, np.array([sample_weights[1]]).T), axis=1)


train.compileModel(learningrate=0.0001, loss=[combined_ccemv, loss_NLL], metrics=['accuracy'],loss_weights=[1., 0.0])


history = model.fit(x=sample_inputs, y=[sample_fldlw,sample_labels[1]], batch_size=100000, epochs=100, validation_split=0.1)


model.save(train.outputDir+"KERAS_model.h5")
pickle.dump(history.history,open(train.outputDir+"Keras_history.p","wb"))


print("completed training")