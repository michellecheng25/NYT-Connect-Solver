import numpy as np

class WordVectors:

    def __init__ (self, lookup, dimensions):
        #lookup is a {word: vector} lookup
        self.lookup, self.dimensions = lookup, dimensions
        #print(self.lookup["ice"])

        #stores words without a vector and total number of words seen
        self.missing, self.total = 0, 0

    def vec(self, word):
        """
        Multiword entries ("ice cream") average the vectors of their pieces.
        Words with no known piece get a zero vector and are counted in self.missing,
        missing word contributes similarity 0 instead of crashing.
        """

        self.total += 1 
        pieces = word.lower().replace("-", "").split()
        vs = [self.lookup[p] for p in pieces if p in self.lookup]
        if not vs:
            self.missing += 1
            return np.zeros(self.dim)
        return np.mean(vs, axis=0)
    
    def load_glove():
        import gensim.downloader as api
        
        kv = api.load("glove-wiki-gigaword-100")
        return WordVectors(kv, 100)


#wv = WordVectors.load_glove()