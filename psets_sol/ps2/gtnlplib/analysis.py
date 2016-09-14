import operator
from gtnlplib.constants import OFFSET
def getTopFeats(weights,class1,class2,allkeys,K=5):
    # your code here
    return sorted([(word, weights[(class1,word)] - weights[(class2,word)]) for word in allkeys if word != OFFSET], key=operator.itemgetter(1))[-1*K:][::-1]
    
