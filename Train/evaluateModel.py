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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpu_options = tf.GPUOptions(allow_growth=True) #,per_process_gpu_memory_fraction=0.1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

from training_base import training_base


train=training_base()

#train.loadModel("/storage/c/dwalter/data/TFModels/DF_2016Boost/KERAS_model.h5")
train.loadModel("/local/scratch/ssd1/dwalter/data/Ntuples_ttbarSelected/180124_all/180207_adv_v3/Base2_1/KERAS_model.h5")
print('model loaded')

#only evaluate on a part...
#n_samples = 500000

#gather the evaluation samples
sample_inputs = train.train_data.getAllFeatures()
sample_labels = train.train_data.getAllLabels()
sample_weights = train.train_data.getAllWeights()

# s_inputs = []
# s_labels = []
# s_weights = []
# for s_input in sample_inputs:
#     s_inputs.append(s_input[:n_samples])
# for s_label in sample_labels:
#     s_labels.append(s_label[:n_samples])
# for s_weight in sample_weights:
#     s_weights.append(s_weight[:n_samples])

#pdb.set_trace()

model = train.keras_model

print('predict ...')
#sample_labels_ID = np.concatenate((sample_labels[0], sample_weights[1]), axis=1)  # attach eventweight to the ID labels

#evaluation = model.evaluate(x=sample_inputs, y=[sample_labels_ID,sample_labels[1]], batch_size=1000)
predictions = model.predict(x=sample_inputs, verbose=1)
#predictions = model.predict(x=sample_inputs, verbose=1)


file = open(train.outputDir+"predictions.p","wb")
pickle.dump([train.train_data.getUsedTruth(), predictions, sample_labels, sample_weights], file)


print("evaluation completed")




