import gtnlplib.constants
from collections import defaultdict
from gtnlplib.constants import END_TAG, START_TAG, TRANS, CURR_SUFFIX, PREV_SUFFIX, OFFSET

def wordFeatures(words,tag,prev_tag,m):
    '''
    :param words: a list of words
    :type words: list
    :param tag: a tag
    :type tag: string
    :type prev_tag: string
    :type m: int
    '''
    out = {(gtnlplib.constants.OFFSET,tag):1}
    if m < len(words): #we can have m = M, for the transition to the end state
        out[(gtnlplib.constants.EMIT,tag,words[m])]=1
    return out

def yourFeatures(words,tag,prev_tag,m):
    output = wordCharFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures

    # your stuff
    #adding prefixes and suffixes of current word and previous word
    CURR_PREFIX = '--curr-pref--'
    PREV_PREFIX = '--prev-pref--'
    NEXT_PREFIX = '--next-pref--'
    NEXT_SUFFIX = '--next-suff--'

    CURR_SUFFIX2 = '--curr-suff2--'
    PREV_SUFFIX2 = '--prev-suff2--'
    CURR_PREFIX2 = '--curr-pref2--'
    PREV_PREFIX2 = '--prev-pref2--'
    NEXT_PREFIX2 = '--next-pref2--'
    NEXT_SUFFIX2 = '--next-suff2--'

    CURR_SUFFIX3 = '--curr-suff3--'
    PREV_SUFFIX3 = '--prev-suff3--'
    CURR_PREFIX3 = '--curr-pref3--'
    PREV_PREFIX3 = '--prev-pref3--'
    NEXT_PREFIX3 = '--next-pref3--'
    NEXT_SUFFIX3 = '--next-suff3--'


    output[(CURR_PREFIX,tag,words[m][0])] = 1
    if len(words[m])>2:
        output[(CURR_PREFIX2,tag,words[m][0:2])] = 1
        output[(CURR_SUFFIX2,tag,words[m][-2:])] = 1
        output[(CURR_PREFIX3,tag,words[m][0:3])] = 1
        output[(CURR_SUFFIX3,tag,words[m][-3:])] = 1
    elif len(words[m])>1:
        output[(CURR_PREFIX2,tag,words[m][0:2])] = 1
        output[(CURR_SUFFIX2,tag,words[m][-2:])] = 1


    if m > 0:
        output[(PREV_PREFIX,tag,words[m-1][0])] = 1
        output[(PREV_SUFFIX,tag,words[m-1][-1])] = 1
        if len(words[m-1]) > 2:
            output[(PREV_PREFIX2, tag, words[m-1][0:2])] = 1
            output[(PREV_SUFFIX2, tag, words[m-1][-2:])] = 1
            output[(PREV_PREFIX3, tag, words[m-1][0:3])] = 1
            output[(PREV_SUFFIX3, tag, words[m-1][-3:])] = 1
        elif len(words[m-1]) > 1:
            output[(PREV_PREFIX2, tag, words[m-1][0:2])] = 1
            output[(PREV_SUFFIX2, tag, words[m-1][-2:])] = 1

    if m < len(words) - 1:
        output[(NEXT_PREFIX,tag,words[m+1][0])] = 1
        output[(NEXT_SUFFIX,tag,words[m+1][-1])] = 1
        if len(words[m+1]) > 2:
            output[(NEXT_PREFIX2, tag, words[m+1][0:2])] = 1
            output[(NEXT_SUFFIX2, tag, words[m+1][-2:])] = 1
            output[(NEXT_PREFIX3, tag, words[m+1][0:3])] = 1
            output[(NEXT_SUFFIX3, tag, words[m+1][-3:])] = 1
        elif len(words[m+1]) > 1:
            output[(NEXT_PREFIX2, tag, words[m+1][0:2])] = 1
            output[(NEXT_SUFFIX2, tag, words[m+1][-2:])] = 1


    return output

def seqFeatures(words,tags,featfunc):
    '''
    :param words: a list of words
    :param tags: a list of tags
    :param featfunc: a function to compute f(words,tag_m,tag_{m-1},m)
    :returns list of features
    '''
    tags.append(END_TAG)
    tags = [START_TAG] + tags
    allfeats = defaultdict(float)
    # your code here
    for i in range(len(words)+1):
        for feat,count in featfunc(words,tags[i+1],tags[i],i).iteritems():
            allfeats[feat] += count 
    return allfeats

def wordCharFeatures(words,tag,prev_tag,m):
    output = wordFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures
    # add your code here
    output[(CURR_SUFFIX,tag,words[m][-1])] = 1
    if m > 0:
        output[(PREV_SUFFIX,prev_tag,words[m-1][-1])] = 1
    return output

def wordTransFeatures(words,tag,prev_tag,m):
    output = wordFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures
    # your code here
    output[(TRANS, tag, prev_tag)] = 1 
    return output

def yourHMMFeatures(words,tag,prev_tag,m):
    output = wordTransFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures

    #add smart stuff
    CURR_PREFIX = '--curr-pref--'
    PREV_PREFIX = '--prev-pref--'

    CURR_SUFFIX2 = '--curr-suff2--'
    CURR_PREFIX2 = '--curr-pref2--'
    PREV_SUFFIX2 = '--prev-suff2--'
    PREV_PREFIX2 = '--prev-pref2--'

    CURR_SUFFIX3 = '--curr-suff3--'
    CURR_PREFIX3 = '--curr-pref3--'
    PREV_SUFFIX3 = '--prev-suff3--'
    PREV_PREFIX3 = '--prev-pref3--'

    if m < len(words) and m > 0:
        output[(PREV_PREFIX, prev_tag, words[m-1][0])] = 1
        output[(PREV_SUFFIX, prev_tag, words[m-1][-1])] = 1
        if len(words[m-1]) > 2:
            output[(PREV_PREFIX2, prev_tag, words[m-1][0:2])] = 1
            output[(PREV_SUFFIX2, prev_tag, words[m-1][-2:])] = 1
            output[(PREV_PREFIX3, prev_tag, words[m-1][0:3])] = 1
            output[(PREV_SUFFIX3, prev_tag, words[m-1][-3:])] = 1
        elif len(words[m-1]) > 1:
            output[(PREV_PREFIX2, prev_tag, words[m-1][0:2])] = 1
            output[(PREV_SUFFIX2, prev_tag, words[m-1][-2:])] = 1
            
    return output
