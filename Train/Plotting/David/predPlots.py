# use the pickled file predictions.p from the evaluateModel.py script do create some output plots

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
from argparse import ArgumentParser
import pdb


parser = ArgumentParser('Make Plots from a KERAS models history')
parser.add_argument('predictions')
#parser.add_argument('pred_data')

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, variance)


args = parser.parse_args()

print('loading pickled data...')

keys= pickle.load(open(args.predictions,'rb'))[0]
predictions = pickle.load(open(args.predictions,'rb'))[1]
truth = pickle.load(open(args.predictions,'rb'))[2]
weights = pickle.load(open(args.predictions,'rb'))[3]

#keys_data = pickle.load(open(args.pred_data,'rb'))[0]
#pred_data = pickle.load(open(args.pred_data,'rb'))[1]
print('got data')


pred_mc = predictions[weights[2]==0]
pred_data = predictions[weights[2]==1]

truth_mc = truth[weights[2]==0]
eventweights_mc = weights[1][weights[2]==0]

#pred_isB = predictions[:,0]
#pred_isBB = predictions[:,1]
#pred_isLeptB = predictions[:,2]
#pred_isC = predictions[:,3]
#pred_isUDS = predictions[:,4]
#pred_isG = predictions[:,5]

pred_mc_isAnyB = pred_mc[:,0]+pred_mc[:,1]+pred_mc[:,2]

pred_data_isAnyB = pred_data[:,0] + pred_data[:,1] + pred_data[:,2]

isB = truth_mc[:,0]
isBB = truth_mc[:,1]
isLeptB = truth_mc[:,2]
isC = truth_mc[:,3]
isUDS = truth_mc[:,4]
isG = truth_mc[:,5]

isAnyB = isB+isBB+isLeptB
isNoB = isC+isUDS+isG
frac_isAnyB = np.sum(isAnyB*eventweights_mc)/np.sum(eventweights_mc)
print("isAnyB fraction = ", frac_isAnyB)
print("isNoB fraction = ", np.sum(isNoB*eventweights_mc)/np.sum(eventweights_mc))

acc_isAnyB = np.sum((np.round(pred_mc_isAnyB)==isAnyB)*eventweights_mc)/np.sum(eventweights_mc)
print("isAnyB accuracy = ", acc_isAnyB)
acc_all = np.sum((np.argmax(pred_mc, axis=1)==np.argmax(truth_mc, axis=1))*eventweights_mc)/np.sum(eventweights_mc)
print("accuracy with all classes = ", acc_all)


#pdb.set_trace()

directory = os.path.dirname('./Plots_Output/')
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

#plot roc curve

fpr, tpr, thresholds = roc_curve(isAnyB, pred_mc_isAnyB, sample_weight=eventweights_mc)
auc_score_mc = roc_auc_score(isAnyB, pred_mc_isAnyB, sample_weight = eventweights_mc)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='isAnyB (auc='+str(auc_score_mc)[:5]+')')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.savefig('roc.png')
plt.cla()


bins = np.arange(0, 1.05, 0.05)
for i,key in enumerate(keys):
    print("key ",i," is ",key)
    av_mc, var_mc = weighted_avg_and_std(pred_mc[:,i], weights=eventweights_mc)
    av_data, var_data = weighted_avg_and_std(pred_data[:,i], weights=np.ones(pred_data.shape[0]))
    plt.hist(pred_mc[:,i], bins,weights = eventweights_mc,  label='mc', histtype='step', normed=True)
    plt.hist(pred_data[:,i], bins,label='data', histtype='step',normed=True)
    plt.text(0.2, plt.ylim()[1] * 0.9, 'mean = ' + str(av_data)[:7]+"(data) "+ str(av_mc)[:7]+"(mc)")
    plt.text(0.2, plt.ylim()[1] * 0.85, 'var = ' + str(var_data)[:7]+"(data) "+ str(var_mc)[:7]+"(mc)")
    plt.text(0.2, plt.ylim()[1] * 0.8, 'mc truth '+str(key)+' fraction = ' + str(np.sum(truth_mc[:,i]*eventweights_mc)/np.sum(eventweights_mc)))
    plt.xlabel('pred_'+key)
    plt.ylabel('frequency')
    plt.legend()
    plt.savefig('pred_'+key+'.png')
    plt.cla()

print("weighted averages are: ")
av_mc, var_mc = weighted_avg_and_std(pred_mc_isAnyB, weights=eventweights_mc)
av_data, var_data = weighted_avg_and_std(pred_data_isAnyB, weights=np.ones(pred_data.shape[0]))

print("mc: ", av_mc)
print("data: ", av_data)

plt.hist(pred_mc_isAnyB,bins,weights = eventweights_mc, label='mc', histtype='step', normed=True )
plt.hist(pred_data_isAnyB, bins, label='data', histtype='step',normed=True)
plt.text(0.2, plt.ylim()[1] * 0.9, 'mean = ' + str(av_data)[:7]+'(data) '+ str(av_mc)[:7]+'(mc)')
plt.text(0.2, plt.ylim()[1] * 0.85, 'var = ' + str(var_data)[:7]+'(data) '+ str(var_mc)[:7]+'(mc)')
plt.text(0.2, plt.ylim()[1] * 0.8, 'mc truth isAnyB fraction = ' + str(frac_isAnyB))
plt.text(0.2, plt.ylim()[1] * 0.75, 'mc truth isAnyB accuracy = ' + str(acc_isAnyB))
plt.xlabel('pred_isAnyB')
plt.ylabel('frequency')
plt.legend()
plt.savefig('pred_isAnyB.png')
plt.cla()



#pdb.set_trace()
