from training_base import training_base

train=training_base(testrun=False)

if not train.modelSet():    # new Model

    from models import model_discriminatorFlat

    train.setModel(model_discriminatorFlat, dropoutRate=0.1)

    train.compileModel(learningrate=0.001,
                       loss=['binary_crossentropy'],
                       metrics=['accuracy']
                       #,loss_weights=[1., 0.000000000001]
                       )


print(train.keras_model.summary())

#Only to prevent tensorflow to take too much resources during training
#import os
#import tensorflow as tf
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#gpu_options = tf.GPUOptions(allow_growth=True
                            #,per_process_gpu_memory_fraction=0.1
#                            )
#s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


model,history = train.trainModel(nepochs=1,
                                 batchsize=500,
                                 stop_patience=300,
                                 lr_factor=0.8,
                                 lr_patience=-3,
                                 lr_epsilon=0.0001,
                                 lr_cooldown=8,
                                 lr_minimum=0.00001,
                                 maxqsize=100)