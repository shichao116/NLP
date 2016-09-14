from sets import Set

def get_token_type_ratio (vocabulary):
    # YOUR CODE HERE
    ntkns = 0
    for key in vocabulary.keys():
        ntkns += vocabulary[key]
    return float(ntkns)/float(len(vocabulary.keys())-1) 

def type_frequency (vocabulary, k):
    # YOUR CODE HERE
    ntypek = 0
    for key,val in vocabulary.items():
        if val==k:
            ntypek += 1
    return ntypek 

def unseen_types (first_vocab, second_vocab):
    # YOUR CODE HERE
    k1 = Set(first_vocab.keys())
    k2 = Set(second_vocab.keys())
    return len(k2-k1) 
