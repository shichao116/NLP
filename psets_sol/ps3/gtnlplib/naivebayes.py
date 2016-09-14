import numpy as np #hint: np.log
from collections import defaultdict, Counter
from gtnlplib.constants import OFFSET
from gtnlplib import scorer

''' keep the shell '''
def learnNBWeights(counts, class_counts, allkeys, alpha=0.1):
    weights = defaultdict(int)
    # your code here
    allcounts = 0

    # get total number of words 
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
            weights[(label, key)] = np.log( float(counts[label][key] + alpha)/float(nwords_label) ) + weights[(label, OFFSET)]
    return weights


