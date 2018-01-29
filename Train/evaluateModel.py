###
# Evaluate a keras model
#   compute predicions on some data
#
###

import pdb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpu_options = tf.GPUOptions(allow_growth=True) #,per_process_gpu_memory_fraction=0.1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

from training_base import training_base
from Losses import loss_NLL, bce_weighted, cce_weighted, bce_weighted_dex, cce_weighted_dex

def combined_bcemv(y_true, y_pred):
    '''
    :param y_true: [isB, isBB, isLeptB, isC, isUDS, isG,  isData, eventweight]
    :param y_pred: [prob_isB, prob_isBB, prob_isLeptB, prob_isC, prob_isUDS, prob_isG]
    :return: combined loss function of cross entropy, means distance and variance distance
    '''

    weight_ce = 1.
    weight_m = 0.5
    weight_v = 0.

    isAnyB = y_true[:,0] + y_true[:,1] + y_true[:,2]
    pred_isAnyB = y_pred[:,0] + y_pred[:,1] + y_true[:,2]
    weight = y_true[:,-1]
    isData = y_true[:,-2]

    #return weight_ce * bce_weighted_dex([isAnyB, isData, weight], pred_isAnyB) +\
    #       weight_m * means_distance([isData, weight], pred_isAnyB) +\
    #       weight_v * variance_distance([isData, weight], pred_isAnyB)


train=training_base()

train.loadModel("/local/scratch/ssd1/dwalter/data/Ntuples_ttbarSelected/180124_all/180129_ccemv_1_01_0/KERAS_model.h5")
#train.loadModel("/storage/c/dwalter/data/TFModels/DF_2016Boost/KERAS_model.h5")
train.compileModel(learningrate=0.0005, loss=[cce_weighted,loss_NLL], metrics=['accuracy'],loss_weights=[1., 0.0])

#gather the evaluation samples
sample_inputs = train.train_data.getAllFeatures()
sample_labels = train.train_data.getAllLabels()
sample_weights = train.train_data.getAllWeights()

#pdb.set_trace()

model = train.keras_model

print('predict ...')

#sample_labels_ID = np.concatenate((sample_labels[0], sample_weights[1]), axis=1)  # attach eventweight to the ID labels

#evaluation = model.evaluate(x=sample_inputs, y=[sample_labels_ID,sample_labels[1]], batch_size=1000)
predictions = model.predict(x=sample_inputs, verbose=1)

file = open(train.outputDir+"predictions.p","wb")
pickle.dump([train.train_data.getUsedTruth(),predictions[0],sample_labels[0],sample_weights],file)


#pdb.set_trace()
print("evaluation completed")




