After previous submission I found the bug leading to low accuracy of viterbi algorithm with
hmm_weights on dev data(58%) ---- i got the order of tuple of transmission weights flipped:
by definition in the note book, it should be hmm_weights[tag(i+1), tag(i), trans], but i 
wrote it as hmm_weights[tag(i), tag(i+1), trans]. That's why the previous version has
accuracy naivebayes.

Here I submit the fixed version below, suffix "_fixed". I hope I won't get penalty for
this less than half hour late. If the score after penalty for the fixed version would be
lower than the original one, please keep the original one

Thanks. 

Chao
