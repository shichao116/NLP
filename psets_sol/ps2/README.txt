DESCRIPTION
===========

This dataset contains text reviews and their labels.
The files you can expect in this dataset are:



1. *.key: A space separated file of atmost two columns: relative location
         of the file containing the text review and the label 
         for the review.
         Expect two files: train.key and dev.key


2. - train/
   - dev/

    These are the directories that contain the text reviews for each set.

3. sentiment-vocab.tff : A lexicon for positive and
                         negative words.


General Notes:

- You should expect that train.key and dev.key have label and a location.
  If there is a missing label or location, something is wrong.

- You should expect that every text review contains some text.

- You should expect that the reviews are not repeated in any of the set.

- You should expect the following statistics about the data

           POS  NEG  NEU TOTAL
  train    590  416  554 1560
  dev      148  111  135 394

- You should expect to see this line in this file! 
