import numpy as np
from collections import Counter
# Implement for deliverable 2a
def CPT (instances, htag):
    """ Accepts instances which is a list and a tag index.
        Computes the conditional probability of modifier given the head tag.

        params:
        instances: list
        htag: integer

        returns:
        output: Dict - where key is a tag and the value is probability.
    """
    counts = Counter() 
    htag_count = 0 
    for inst in instances:
        for m in range(1, len(inst.pos)):
            if inst.pos[inst.heads[m]] == htag:
                counts[(inst.pos[m], htag)] += 1
                htag_count += 1
    
    output = {}
    total_count = 0
    for key, val in counts.iteritems():
        total_count += val
        output[key] = float(val)/htag_count
    assert total_count == htag_count
    return output


def entropy (distr):
    """ Calculates the entropy of a given distribution """
    return np.sum(np.array(distr.values())*np.log2(distr.values()))
