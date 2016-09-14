from gtnlplib import scorer
from gtnlplib import preproc
from gtnlplib import clf_base
from gtnlplib.constants import DEV_FILE, OFFSET, TRAIN_FILE
from collections import defaultdict, Counter
import operator
import numpy as np
argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]
argmin = lambda x : min(x, key=operator.itemgetter(1))[0]

def predict(weights, word, tags):
    seen = False
    for tag in tags:
        seen = seen or weights.has_key((tag,word))
    if not seen:
        offset_weights = [(tag, weights[(tag,OFFSET)]) for tag in tags if weights.has_key((tag,OFFSET))]
        return argmin(offset_weights) 

    scores = defaultdict(int)
    for tag in tags:
        scores[tag] = weights[(tag,word)]
    return argmax(scores)

def makeClassifierTagger(weights):
    #Code here
    tagger = lambda words, alltags: [predict(weights, word, alltags) for word in words] 
    return tagger 

def evalTagger(tagger,outfilename,testfile=DEV_FILE):
    """Calculate confusion_matrix for a given tagger

    Parameters:
    tagger -- Function mapping (words, possible_tags) to an optimal
              sequence of tags for the words
    outfilename -- Filename to write tagger predictions to
    testfile -- (optional) Filename containing true labels

    Returns:
    confusion_matrix -- dict of occurences of (true_label, pred_label)
    """
    alltags = set()
    for i,(words, tags) in enumerate(preproc.conllSeqGenerator(TRAIN_FILE)):
        for tag in tags:
            alltags.add(tag)
    with open(outfilename,'w') as outfile:
        for words,_ in preproc.conllSeqGenerator(testfile):
            pred_tags = tagger(words,alltags)
            for tag in pred_tags:
                print >>outfile, tag
            print >>outfile, ""
    return scorer.getConfusion(testfile,outfilename) #run the scorer on the prediction file
