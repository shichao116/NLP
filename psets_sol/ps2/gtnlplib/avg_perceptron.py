import operator
from  constants import *
from collections import defaultdict, Counter
from clf_base import predict, evalClassifier
import scorer

def trainAvgPerceptron(N_its,inst_generator,labels, outfile, devkey):
    tr_acc = [None]*N_its
    dv_acc = [None]*N_its
    weights = defaultdict(float)
    wsum = defaultdict(float)
    for i in xrange(N_its):
        weights, wsum, tr_err, num_insts = oneItAvgPerceptron(inst_generator, weights, wsum, labels, i+1)
        avg_weights = weights.copy()
        for key in wsum.keys():
            avg_weights[key] -= 1/float(i+2)*wsum[key]
        confusion = evalClassifier(avg_weights,outfile, devkey) #evaluate on dev data
        dv_acc[i] = scorer.accuracy(confusion) #compute accuracy
        tr_acc[i] = 1. - tr_err/float(num_insts) #compute training accuracy from output
        print i,'dev: ',dv_acc[i],'train: ',tr_acc[i]
    return avg_weights, tr_acc, dv_acc


def oneItAvgPerceptron(inst_generator,weights,wsum,labels,Tinit=0):
    errors = 0.
    num_insts = float(len(inst_generator))

    for inst in inst_generator:
        label, _ = predict(inst[0], weights, labels)
        if label != inst[1]:
            errors += 1
            for word, count in inst[0].items():
                weights[(inst[1], word)] += count
                weights[(label, word)] -= count
                wsum[(inst[1],word)] += Tinit*count
                wsum[(label,word)] -= Tinit*count
    return weights, wsum, errors, num_insts 
