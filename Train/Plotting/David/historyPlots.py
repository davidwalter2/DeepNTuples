# use the pickled file history.p from the train_deepFlavour_da.py script do create some plots about the metrics and losses during the training

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import pickle, pdb

parser = ArgumentParser('Make Plots from a KERAS models history')
parser.add_argument('history')

args = parser.parse_args()

history = pickle.load(open(args.history,"rb"))



#pdb.set_trace()

def plotHistory(key, validation = False):
    plt.plot(history[key], label='training')
    plt.plot(history["val_"+key], label = 'validation')
    plt.xlabel('epoch')
    plt.ylabel(key)
    plt.legend()
    # plt.show()
    plt.savefig(key + '.png')
    plt.cla()

### Main part

directory = os.path.dirname('./Plots_History/')
# make a canvas, draw, and save it
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

val = False
if "val_" in history.keys()[-1]:
    val = True

for key in history.keys():
    if not key.startswith("val_"):
        plotHistory(key, validation = val)


