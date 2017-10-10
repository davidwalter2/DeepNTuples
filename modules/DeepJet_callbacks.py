'''
Created on 7 Apr 2017

@author: jkiesele
'''
from __future__ import print_function

from ReduceLROnPlateau import ReduceLROnPlateau
from keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint, TensorBoard, LambdaCallback #, ReduceLROnPlateau #
# loss per epoch
from time import time
from pdb import set_trace
import json

class newline_callbacks_begin(Callback):
    
    def __init__(self,outputDir):
        self.outputDir=outputDir
        self.loss=[]
        self.val_loss=[]
        self.full_logs=[]
        
    def on_epoch_end(self,epoch, epoch_logs={}):
        import os
        lossfile=os.path.join( self.outputDir, 'losses.log')
        print('\n***callbacks***\nsaving losses to '+lossfile)
        self.loss.append(epoch_logs.get('loss'))
        self.val_loss.append(epoch_logs.get('val_loss'))
        f = open(lossfile, 'a')
        f.write(str(epoch_logs.get('loss')))
        f.write(" ")
        f.write(str(epoch_logs.get('val_loss')))
        f.write("\n")
        f.close()    
        normed = {}
        for vv in epoch_logs:
            normed[vv] = float(epoch_logs[vv])
        self.full_logs.append(normed)
        lossfile=os.path.join( self.outputDir, 'full_info.log')
        with open(lossfile, 'w') as out:
            out.write(json.dumps(self.full_logs))
        
class newline_callbacks_end(Callback):
    def on_epoch_end(self,epoch, epoch_logs={}):
        print('\n***callbacks end***\n')
        
        
class Losstimer(Callback):
    def __init__(self, every = 5):
        self.points = []
        self.every = every

    def on_train_begin(self, logs):
        self.start = time()

    def on_batch_end(self, batch, logs):
        if (batch % self.every) != 0: return
        elapsed = time() - self.start
        cop = {}
        for i, j in logs.iteritems():
            cop[i] = float(j)
        cop['elapsed'] = elapsed
        self.points.append(cop)
        
        
class checkTokens_callback(Callback):
    
    def __init__(self,cutofftime_hours=48):
        self.cutofftime_hours=cutofftime_hours
        
    def on_epoch_begin(self, epoch, logs=None):
        from tokenTools import checkTokens
        checkTokens(self.cutofftime_hours)
        
class saveCheckPointDeepJet(Callback):
    '''
    this seems obvious, however for some reason the keras model checkpoint fails
    to save the optimizer state, needed for resuming a training. Therefore this explicit
    implementation.
    '''
    
    def __init__(self,outputDir,model):
        self.outputDir=outputDir
        self.djmodel=model
    def on_epoch_end(self,epoch, epoch_logs={}):
        self.djmodel.save(self.outputDir+"/KERAS_check_model_last.h5")

#under construction
class predictionHistory(Callback):

    def __init__(self,outputDir, model):
        Callback.__init__(self)
        self.djmodel = model
        self.outputDir = outputDir

    def on_epoch_end(self, batch, logs={}):
        y_pred = self.model.predict(self.model.training_data)
        y_true = self.model.validation_data[1]

        import os
        predhisfile=os.path.join( self.outputDir, 'predhis.log')
        f = open(predhisfile, 'a')
        f.write(str(y_pred))
        f.write(" ")
        f.write(str(y_true))

        f.write("\n")
        f.close()
        return
        
        
class DeepJet_callbacks(object):
    def __init__(self,
                 model,
                 datacollection,
                 stop_patience=10,
                 lr_factor=0.5,
                 lr_patience=1,
                 lr_epsilon=0.001,
                 lr_cooldown=4,
                 lr_minimum=1e-5,
                 outputDir='',
                 minTokenLifetime=5,
                 checkperiod=10):
        

        
        self.nl_begin=newline_callbacks_begin(outputDir)
        self.nl_end=newline_callbacks_end()
        
        self.stopping = EarlyStopping(monitor='val_loss', 
                                      patience=stop_patience, 
                                      verbose=1, mode='min')
        
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, 
                                mode='min', verbose=1, epsilon=lr_epsilon,
                                 cooldown=lr_cooldown, min_lr=lr_minimum)

        self.modelbestcheck=ModelCheckpoint(outputDir+"/KERAS_check_best_model.h5", 
                                        monitor='val_loss', verbose=1, 
                                        save_best_only=True, save_weights_only=False)
        

        self.modelcheckperiod=ModelCheckpoint(outputDir+"/KERAS_check_model_epoch{epoch:02d}.h5", verbose=1,period=checkperiod, save_weights_only=False)

        
        self.modelcheck=saveCheckPointDeepJet(outputDir,model)

        self.tensorboard = TensorBoard(log_dir=outputDir+"/logs",histogram_freq=1)
  
        self.history = History()
        self.timer = Losstimer()

        self.predictionhistory = predictionHistory(outputDir, model)
        
        self.tokencheck=checkTokens_callback(minTokenLifetime)


        def pred(epoch, logs):
            predx, predy = next(datacollection.generator())

            predout = model.predict(
                predx,
                batch_size=1
            )

            print("Target\n")
            print(predy)
            print("Prediction\n")
            print(predout)

        self.pred = LambdaCallback(on_batch_end=pred)

  
        self.callbacks=[
            self.nl_begin, self.tokencheck,
            self.modelbestcheck, self.modelcheck,self.modelcheckperiod,
            self.reduce_lr, self.stopping, self.nl_end, self.history,
            self.timer
            #, self.tensorboard, self.testmodelcb
        ]
