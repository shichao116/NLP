# You may store the dataset anywhere you want.
# Make the appropriate changes to the following constants.
from os.path import expanduser

home = expanduser("~")
TRAINKEY = home+'/Dropbox/course/gt-nlp-class/psets_sol/ps2/train.key'
DEVKEY = home+'/Dropbox/course/gt-nlp-class/psets_sol/ps2/dev.key'
TESTKEY = home+'/Dropbox/course/gt-nlp-class/psets_sol/ps2/test.key'

# Sentiment file - should be present in the dataset
SENTIMENT_FILE = home+"/Dropbox/course/gt-nlp-class/psets_sol/ps2/sentiment-vocab.tff"

OFFSET = '**OFFSET**'

# A list of labels.
ALL_LABELS= ["POS", "NEG", "NEU"]
