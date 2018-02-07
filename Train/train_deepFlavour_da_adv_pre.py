###For the pre-training of the discriminator of an adversarial net


from training_base import training_base
from models import model_deepFlavourReference_gradientReversal
from Losses import loss_NLL, bce_weighted, cce_weighted_dex
from modelTools import fixLayersContaining

import os
import pdb
import numpy as np
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train=training_base()

train.setModel(model_deepFlavourReference_gradientReversal)

model = train.keras_model


#load the weights from the pretrained part of the model
model.load_weights("/storage/c/dwalter/data/TFModels/DF_2016Boost/KERAS_model.h5", by_name=True)

fixLayersContaining(model,"discriminator",invert=True)

#pdb.set_trace()

#gather the training samples
sample_inputs = train.train_data.getAllFeatures()
sample_labels = train.train_data.getAllLabels()
sample_weights = train.train_data.getAllWeights()

#shuffle data one time before fit, because validation data has to be shuffled too, this takes some time ...
print('shuffle data ...')
from Helpers import shuffle_in_unison
sample_inputs[0], sample_inputs[1], sample_inputs[2], sample_inputs[3], sample_inputs[4], sample_labels[0], sample_labels[1], sample_weights[0], sample_weights[1], sample_weights[2] = shuffle_in_unison((sample_inputs[0], sample_inputs[1], sample_inputs[2], sample_inputs[3], sample_inputs[4], sample_labels[0], sample_labels[1], sample_weights[0], sample_weights[1], sample_weights[2]))
print('done shuffling data')

sample_fldlw = np.concatenate((sample_labels[0], np.array([sample_weights[2]]).T, np.array([sample_weights[1]]).T), axis=1)
sample_dlw = np.concatenate((np.array([sample_weights[2]]).T, np.array([sample_weights[1]]).T), axis=1)


#Model outputs are flavour prediction, regression prediction and domain prediction
train.compileModel(learningrate=0.0001, loss=[cce_weighted_dex, loss_NLL, bce_weighted],loss_weights=[0., 0., 1.])
#train.compileModel(learningrate=0.0001, loss=["categorical_crossentropy", loss_NLL, "binary_crossentropy"],loss_weights=[0., 0., 1.])

#first train only the discriminator

history = model.fit(x=sample_inputs, y=[sample_fldlw, sample_labels[1], sample_dlw], batch_size=100000, epochs=10, validation_split=0.1)
#history = model.fit(x=sample_inputs, y=[sample_labels[0], sample_labels[1], sample_weights[2]],  batch_size=20000, epochs=10, validation_split=0.1,sample_weight=[sample_weights[1],sample_weights[1],sample_weights[1]])


#pdb.set_trace()
model.save(train.outputDir+"KERAS_model.h5")
pickle.dump(history.history,open(train.outputDir+"Keras_history.p","wb"))

#pdb.set_trace()

