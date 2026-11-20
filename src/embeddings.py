import numpy as np

class Embedding:

    def __init__(self):
        self.missing, self.total = 0, 0

    def vectors(self, words):
        return np.array([self.vec(w) for w in words])

    def _average(self, keys):
        """
        for two words, if an embedding for it doesnt exist, then average the two embeddings
        e.g. ice cream -> average of ice and cream
        """
        vs = [self.kv[k] for k in keys if k in self.kv]
        if not vs:
            self.missing += 1
            return np.zeros(self.dim)
        return np.mean(vs, axis=0)


class GloVe(Embedding):
    def __init__(self, model="glove-wiki-gigaword-100"):
        super().__init__()
        import gensim.downloader as api

        self.kv = api.load(model)
        self.dim = self.kv.vector_size

    def vec(self, word):
        self.total += 1
        text = word.lower().replace("-", " ").strip()
        # "ice cream" -> average of ice, cream
        return self._average(text.split())


class Numberbatch(Embedding):
    def __init__(self, path="data/numberbatch-en-19.08.txt.gz", cache="data/numberbatch.kv"):
        super().__init__()
        import os
        from gensim.models import KeyedVectors

        # reading the .txt.gz takes about a minute, so the first load saves a binary copy;
        # after that, loading the copy takes under a second
        if os.path.exists(cache):
            self.kv = KeyedVectors.load(cache, mmap="r")
        else:
            self.kv = KeyedVectors.load_word2vec_format(path, binary=False)
            self.kv.save(cache)
        self.dim = self.kv.vector_size

    def vec(self, word):
        self.total += 1
        parts = word.lower().replace("-", " ").split()
        #for two words: ice cream, it becomes "ice_cream"
        for key in ("_".join(parts), "".join(parts)):
            if key in self.kv:
                return np.asarray(self.kv[key])
        # otherwise average the pieces
        return self._average(parts)



EMBEDDINGS = {"glove": GloVe, "numberbatch": Numberbatch}


def get_embedding(name):
    return EMBEDDINGS[name]()