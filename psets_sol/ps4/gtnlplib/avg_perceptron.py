from collections import defaultdict
from gtnlplib.tagger_base import classifierTagger
from gtnlplib.tagger_base import evalTagger 
from gtnlplib import scorer

def oneItAvgPerceptron(inst_generator,featfunc,weights,wsum,tagset,Tinit=0):
    """
    :param inst_generator: iterator over instances
    :param featfunc: feature function on (words, tag_m, tag_m_1, m)
    :param weights: default dict
    :param wsum: weight sum, for averaging
    :param tagset: set of permissible tags
    :param Tinit: initial value of t, the counter over instances
    """
    tr_err = 0.0
    for i,(words,y_true) in enumerate(inst_generator):
        # your code here
        y_pred= classifierTagger(words, featfunc, weights, tagset)
        assert len(y_true) == len(y_pred)
        for j in range(len(y_true)):
            if y_true[j] != y_pred[j]:
                tr_err += 1
                for feat,count in featfunc(words, y_true[j],'DUMMY',j).iteritems():
                    weights[feat] += count 
                    wsum[feat] += (Tinit+i)*count
                for feat,count in featfunc(words,y_pred[j],'DUMMY',j).iteritems():
                    weights[feat] -= count 
                    wsum[feat] -= (Tinit+i)*count
                #if y_true[j] == '^' and y_pred[j] == 'N':
                #    print 'error type 1', words, y_true, y_pred
                #if y_true[j] == 'V' and y_pred[j] == 'N':
                #    print 'error type 2', words, y_true, y_pred
                #if y_true[j] == 'A' and y_pred[j] == 'N':
                #    print 'error type 3', words, y_true, y_pred
    # note that i'm computing tr_acc for you, as long as you properly update tr_err
    return weights, wsum, 1.-tr_err / float(sum([len(s) for s,t in inst_generator])), i


def trainAvgPerceptron(N_its,inst_generator,featfunc,tagset):
    """
    :param N_its: number of iterations
    :param inst_generator: generate words,tags pairs
    :param featfunc: feature function
    :param tagset: set of all possible tags
    :returns average weights, training accuracy, dev accuracy
    """
    tr_acc = [None]*N_its
    dv_acc = [None]*N_its
    T = 0
    avg_weights = defaultdict(float)
    weights = defaultdict(float)
    wsum = defaultdict(float)
    for i in xrange(N_its):
        # your code here
        weights, wsum, tr_acc_i, num_insts = oneItAvgPerceptron(inst_generator, featfunc, weights, wsum, tagset, i)
        avg_weights = weights.copy()
        T += num_insts
        for key in wsum.keys():
            #avg_weights[key] -= 1/float(i+1)*wsum[key]
            avg_weights[key] -= 1/T*wsum[key]
        confusion = evalTagger(lambda words, alltags: classifierTagger(words,featfunc,avg_weights,tagset),'perc')
        dv_acc[i] = scorer.accuracy(confusion)
        tr_acc[i] = tr_acc_i
        print i,'dev:',dv_acc[i],'train:',tr_acc[i]
    return avg_weights, tr_acc, dv_acc
