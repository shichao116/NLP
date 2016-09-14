''' your code '''
import operator
from  constants import *
from collections import defaultdict, Counter
from gtnlplib import preproc
import scorer
from gtnlplib import constants
from gtnlplib import clf_base
import numpy as np
argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

def get_tags(trainfile):
    """Produce a Counter of occurences of word in each tag"""
    counters = defaultdict(Counter)
    for i, (words, tags) in enumerate(preproc.conllSeqGenerator(trainfile)):
        for j in range(len(words)):
            counters[tags[j]][words[j]] += 1  
    return counters

def get_noun_weights():
    """Produce weights dict mapping all words as noun"""
    weights_all_noun = Counter() 
    weights_all_noun.update({('N',constants.OFFSET):1})
    return weights_all_noun 

def get_most_common_weights(trainfile):
    counters = get_tags(trainfile)
    counts = get_class_counts(counters)
    weights = defaultdict(int)
    for tag in counters.keys():
        weights[(tag,constants.OFFSET)] = counts[tag] 
        for word, count in counters[tag].iteritems():
            weights[(tag,word)] = count
    return weights

def get_class_counts(counters):
    counts = Counter()
    for tag, counter in counters.iteritems():
        for word in counter.keys():
            counts[tag] += counter[word]
    return counts


