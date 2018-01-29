### collection of prediction plots of different predictions (of different models)
#   needs predictions.p file from evaluateModel.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
from argparse import ArgumentParser
import pdb



class Source:
    """ DOC """

    collection = []

    def __init__(self, source, name = ''):

        self.keys = pickle.load(open(source, 'rb'))[0]
        self.predictions = pickle.load(open(source, 'rb'))[1]
        self.truth = pickle.load(open(source, 'rb'))[2]
        self.weights = pickle.load(open(source, 'rb'))[3]

        self.name = name

        self.pred_mc = self.predictions[self.weights[2] == 0]
        self.pred_data = self.predictions[self.weights[2] == 1]

        self.truth_mc = self.truth[self.weights[2] == 0]
        self.eventweights_mc = self.weights[1][self.weights[2] == 0]

        self.pred_mc_isAnyB = self.pred_mc[:, 0] + self.pred_mc[:, 1] + self.pred_mc[:, 2]

        self.pred_data_isAnyB = self.pred_data[:, 0] + self.pred_data[:, 1] + self.pred_data[:, 2]

        self.isB = self.truth_mc[:, 0]
        self.isBB = self.truth_mc[:, 1]
        self.isLeptB = self.truth_mc[:, 2]
        self.isC = self.truth_mc[:, 3]
        self.isUDS = self.truth_mc[:, 4]
        self.isG = self.truth_mc[:, 5]

        self.isAnyB = self.isB + self.isBB + self.isLeptB
        self.isNoB = self.isC + self.isUDS + self.isG

        self.fpr, self.tpr, self.thresholds = roc_curve(self.isAnyB, self.pred_mc_isAnyB, sample_weight=self.eventweights_mc)
        self.auc_score_mc = roc_auc_score(self.isAnyB, self.pred_mc_isAnyB, sample_weight=self.eventweights_mc)

        Source.collection.append(self)


    def plot_roc(self):

        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(self.fpr, self.tpr, label=self.name+' (auc=' + str(self.auc_score_mc)[:5] + ')')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('isAnyB ROC curve')
        plt.legend()
        plt.savefig('roc.png')
        plt.cla()

    @classmethod
    def plot_rocs(cls):
        plt.plot([0, 1], [0, 1], 'k--')
        for src in cls.collection:
            plt.plot(src.fpr, src.tpr, label=src.name+' (auc=' + str(src.auc_score_mc)[:6] + ')')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('isAnyB ROC curve')
        #plt.xscale('log')
        plt.legend()
        plt.savefig('roc.png')
        plt.cla()




pred_pre = Source('/local/scratch/ssd1/dwalter/data/Ntuples_ttbarSelected/180124_all/Predictions/predictions.p',name='pretrained')
pred_180125ToF = Source('/local/scratch/ssd1/dwalter/data/Ntuples_ttbarSelected/180124_all/180125_TrainedOnFlavour/Predictions/predictions.p', name='180125ToF')
pred_180126ccemv = Source('/local/scratch/ssd1/dwalter/data/Ntuples_ttbarSelected/180124_all/180126_Test_ccemv/Predictions/predictions.p', name='180126ccemv')

print("make directory and save plots")
directory = os.path.dirname('./plots_rocCollection/')
# make a canvas, draw, and save it
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

Source.plot_rocs()

