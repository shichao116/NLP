import numpy as np #hint: np.log
from itertools import chain
from collections import defaultdict, Counter
from gtnlplib.preproc import dataIterator
from gtnlplib.constants import OFFSET, TRAINKEY, DEVKEY
from gtnlplib import scorer
from gtnlplib.clf_base import evalClassifier

''' keep the shell '''
def learnNBWeights(counts, class_counts, allkeys, alpha=0.1):
    weights = defaultdict(int)
    # your code here
    allcounts = 0

    # get total number of documents
    for label, val in class_counts.items():
        allcounts += val 
    
    # get prior, i.e. \mu, logP(y)
    for label, val in class_counts.items():
        weights[(label,OFFSET)] = np.log(float(val)/float(allcounts))

    # get \phi_{j,n}
    for label in class_counts.keys():
        nwords_label = (len(allkeys)-1)*alpha
        for key in allkeys:
            if key == OFFSET:
                continue
            nwords_label += counts[label][key]
        for key in allkeys:
            if key == OFFSET:
                continue
            weights[(label, key)] = np.log( float(counts[label][key] + alpha)/float(nwords_label) )

    return weights

def regularization_using_grid_search (alphas, counts, class_counts, allkeys, tr_outfile='nb.alpha.tr.txt', dv_outfile='nb.alpha.dv.txt'):
    tr_accs = []
    dv_accs = []
    # Choose your alphas here
    weights_nb_alphas = dict()
    for alpha in alphas:
        weights_nb_alphas[alpha] = learnNBWeights(counts, class_counts, allkeys, alpha)
        confusion = evalClassifier(weights_nb_alphas[alpha],tr_outfile,TRAINKEY)
        tr_accs.append(scorer.accuracy(confusion))
        confusion = evalClassifier(weights_nb_alphas[alpha],dv_outfile,DEVKEY)
        dv_accs.append(scorer.accuracy(confusion))
    return weights_nb_alphas, tr_accs, dv_accs
