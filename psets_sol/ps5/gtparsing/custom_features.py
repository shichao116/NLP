from dependency_features import DependencyFeatures
from numpy import sign
import numpy as np
class LexFeats(DependencyFeatures):
    def create_arc_features(self,instance,h,m,add=False):
        """ Notes about the code
        - You start by calling the same function, using the parent class. 
          You can build a chain of feature functions in this way.
        - h provides the index of the head word of the dependency arc
        - m provides the index of the modifier word of the dependency arc
        - You can access the part of speech tags in the instance as instance.pos[i], 
          where i indexes any word token.
        - You can access the words themselves as instance.words[i], 
          where i again indexes the token
        - To create a feature, you call getF(), with two arguments:
          - A feature tuple, which includes an index k, and any other information 
            you want -- it need not be a tuple of exactly three items
          - An argument "add", which you don't need to worry about 
            (but you do need to include)
          - Make sure to keep k up-to-date. 
            This prevents collisions in the space of features.
        """
        ff = super(LexFeats,self).create_arc_features(instance,h,m,add)
        k = len(ff)
        f = self.getF((k,instance.pos[h],instance.words[m]),add)
        ff.append(f)
        return ff


# For Deliverable 1a
class LexDistFeats(LexFeats):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(LexDistFeats,self).create_arc_features(instance,h,m,add)
        k = len(ff)
        dist = h - m
        if abs(dist) > 10:
            dist = sign(dist)*10
        f = self.getF((k,dist),add)
        ff.append(f)
        return ff 

# For Deliverable 1b
class LexDistFeats2(LexDistFeats):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(LexDistFeats2,self).create_arc_features(instance,h,m,add)
        k = len(ff)
        f = self.getF((k,instance.words[h],instance.pos[m]),add)
        ff.append(f)
        return ff 

# For Deliverable 1c
class ContextFeats(LexDistFeats2):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(ContextFeats,self).create_arc_features(instance,h,m,add)
        k = len(ff)
        if h > 0:
            f = self.getF((k,instance.pos[h],instance.pos[h-1],instance.pos[m]),add)
            ff.append(f)
            k = len(ff)
        if m < np.size(instance.words)-1:
            f = self.getF((k,instance.pos[h],instance.pos[m],instance.pos[m+1]),add)
            ff.append(f)
            k = len(ff)
        if h > 0 and m < np.size(instance.words)-1:
            f = self.getF((k,instance.pos[h],instance.pos[h-1], instance.pos[m], instance.pos[m+1]),add)
            ff.append(f)
        return ff 

# For Deliverable 2c
class DelexicalizedFeats(DependencyFeatures):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = []
        dist = h - m
        if np.abs(dist) > 10:
            dist = np.sign(dist)*10
        f = self.getF((0,instance.pos[h],instance.pos[m],dist),add)
        ff.append(f)
        return ff

# For Deliverable 2e
class RelexicalizedFeats(DelexicalizedFeats):
    def create_arc_features(self, instance, h, m, add=False):
        ff = super(RelexicalizedFeats, self).create_arc_features(instance,h,m,add)
        k = len(ff)
        f = self.getF((k,instance.pos[h],instance.words[m]),add)
        ff.append(f)
        return ff

# For Deliverable 2f
class CntxtMorphFeats(RelexicalizedFeats):
    def create_arc_feature(self, instance, h, m, add=False):
        ff = super(CntxtMorphFeats, self).create_arc_features(instance,h,m,add)
        k = len(ff)
        # context features, same as in class ContextFeats
        if h > 0:
            f = self.getF((k,instance.pos[h],instance.pos[h-1],instance.pos[m]),add)
            ff.append(f)
            k = len(ff)
        if m < np.size(instance.words)-1:
            f = self.getF((k,instance.pos[h],instance.pos[m],instance.pos[m+1]),add)
            ff.append(f)
            k = len(ff)
        if h > 0 and m < np.size(instance.words)-1:
            f = self.getF((k,instance.pos[h],instance.pos[h-1], instance.pos[m], instance.pos[m+1]),add)
            ff.append(f)
            k = len(ff)

        # morphological features:
        word_h = self.word_dict[instance.words[h]]
        word_m = self.word_dict[instance.words[m]]
        if len(word) > 1:
            f = self.getF((k,instance.pos[h],word_h[-2:]),add)
            ff.append(f)
            k = len(ff)
            if len(word_m) > 1:
                f = self.getF((k,word_h[-2:],word_m[-2:],word_h[:2],word_m[:2]),add)
                k = len(ff)
        return ff
        

    










