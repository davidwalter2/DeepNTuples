from training_base import training_base
from Losses import loss_NLL, bce_weighted_dex, cce_weighted_dex, cce_weighted, moments_weighted, metric_means, metric_variances
from modelTools import fixLayersContaining
import tensorflow as tf

import pdb
import numpy as np
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#gpu_options = tf.GPUOptions(allow_growth=True) #,per_process_gpu_memory_fraction=0.1)
#s = tf.InteractiveSession(config=tf.ConfigProto())#gpu_options=gpu_options))


def combined_ccemv(y_true, y_pred):
    '''
    :param y_true: [isB, isBB, isLeptB, isC, isUDS, isG,  isData, eventweight]
    :param y_pred: [prob_isB, prob_isBB, prob_isLeptB, prob_isC, prob_isUDS, prob_isG]
    :return: combined loss function of cross entropy and moments (means and variances)
    '''

    weight_ce = 1.
    weight_m = 0.2
    weight_v = 0.04

    y_true_dlw =  y_true[:,-2:]  #domain label and eventweight

    return moments_weighted(y_true_dlw,y_pred,[weight_m,weight_v]) + weight_ce * cce_weighted_dex(y_true, y_pred)



def shuffle_in_unison(listofArrays):
    '''
    :param list of arrays: a list of numpy arrays with the same length in the first dimension
    :return: list of arrays shuffled only in the first dimension
    '''
    shuffled_list = []
    for arr in listofArrays:
        shuffled_list.append(np.empty(arr.shape, dtype=arr.dtype))
    permutation = np.random.permutation(len(listofArrays[0]))
    for old_index, new_index in enumerate(permutation):
        for i, arr in enumerate(listofArrays):
            shuffled_list[i][new_index] = arr[old_index]
    return shuffled_list

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

#pdb.set_trace()

train.compileModel(learningrate=0.0001, loss=[combined_ccemv, loss_NLL], metrics={'ID_pred':[cce_weighted_dex, metric_means, metric_variances]},loss_weights=[1., 0.0])


#gather the training samples
sample_inputs = train.train_data.getAllFeatures()
sample_labels = train.train_data.getAllLabels()
sample_weights = train.train_data.getAllWeights() #[weight, eventweight, domainLabel]


#shuffle data one time before fit, because validation data has to be shuffled too, this takes some time ...
print('shuffle data ...')
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



history = model.fit(x=sample_inputs, y=[sample_fldlw,sample_labels[1]], batch_size=100000, epochs=1, validation_split=0.1)

#recompile without metrics, they would cause trouble when loading the model
train.compileModel(learningrate=0.0001, loss=[combined_ccemv, loss_NLL],loss_weights=[1., 0.0])


model.save(train.outputDir+"KERAS_model.h5")
pickle.dump(history.history,open(train.outputDir+"Keras_history.p","wb"))


print("completed training")