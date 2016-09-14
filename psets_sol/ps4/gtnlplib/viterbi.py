import numpy as np #hint: np.log
from itertools import chain
import operator
from collections import defaultdict, Counter
from gtnlplib.preproc import conllSeqGenerator

from gtnlplib import scorer
from gtnlplib import constants
from gtnlplib import preproc
from gtnlplib.constants import START_TAG ,TRANS ,END_TAG , EMIT

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

# define viterbiTagger
start_tag = constants.START_TAG
trans = constants.TRANS
end_tag = constants.END_TAG
emit = constants.EMIT


def viterbiTagger(words,feat_func,weights,all_tags,debug=False):
    """
    :param words: list of words
    :param feat_func: feature function
    :param weights: defaultdict of weights
    :param tagset: list of permissible tags
    :param debug: optional debug flag
    :returns output: tag sequence
    :returns best_score: viterbi score of best tag sequence
    """
    trellis = [None] * len(words)
    pointers = [None] * len(words)
    output = [None] * len(words)
    for i in range(len(words)):
        pointer = defaultdict(str) 
        scores = defaultdict(float) 
        for curr_tag in all_tags:
            if i == 0:
                prev_tag = START_TAG 
                pointer[curr_tag] = prev_tag 
                #feat = feat_func(words,curr_tag,prev_tag,i)
                for feat in feat_func(words, curr_tag, prev_tag, i):
                    scores[curr_tag] += weights[feat]
            else:
                max_score= -np.inf
                ptag = None
                for prev_tag in all_tags:
                    #feat = feat_func(words,curr_tag,prev_tag,i)
                    temp_score = trellis[i-1][prev_tag]
                    for feat in feat_func(words, curr_tag, prev_tag, i):
                        temp_score += weights[feat]
                    if max_score < temp_score:
                        max_score = temp_score
                        ptag = prev_tag
                    #if max_score < trellis[i-1][prev_tag] + weights[feat[0]] + weights[feat[1]]:
                    #    max_score = trellis[i-1][prev_tag] + weights[feat[0]] + weights[feat[1]]
                    #    ptag = prev_tag
                pointer[curr_tag] = ptag
                scores[curr_tag] = max_score
        trellis[i] = scores
        pointers[i] = pointer

    #deal with the last word
    curr_tag = END_TAG
    max_score = -np.inf
    ptr_last = None
    for prev_tag in all_tags:
        #feat = feat_func(words,curr_tag,prev_tag, len(words))
        temp_score = trellis[len(words)-1][prev_tag]
        for feat in feat_func(words, curr_tag, prev_tag, len(words)):
            temp_score += weights[feat]
        if max_score < temp_score:
            max_score = temp_score
            ptr_last = prev_tag
        #if max_score < trellis[len(words)-1][prev_tag] + weights[feat[0]]:
        #    max_score = trellis[len(words)-1][prev_tag] + weights[feat[0]]
        #    ptr_last = prev_tag 
    best_score = max_score
    output = [ptr_last]
    prev_tag = ptr_last
    for i in range(len(words))[::-1]:
        output.append(pointers[i][prev_tag])
        prev_tag = pointers[i][prev_tag]
    output = output[::-1][1:]
    return output,best_score


