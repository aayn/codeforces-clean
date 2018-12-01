import numpy as np
from torch.utils.data import Dataset
from . import vectorizer as vz


class TokenDataset(Dataset):
    "Similar to fastai TextDataset."
    def __init__(self, X, y, backwards=False, sos=None, eos=None,
                 onehot=False, tvectorizer=None, ovectorizer=None,
                 svectorizer=None):
        self.backwards, self.sos, self.eos = backwards, sos, eos
        self.tokens = None
        if tvectorizer is None:
            self.tvectorizer = vz.TokenVectorizer()
            self.X, self.tokens = self.tvectorizer.fit_transform(X)
        else:
            self.tvectorizer = tvectorizer
            self.X, self.tokens = self.tvectorizer.transform(X)

        if onehot is True:
            self.svectorizer = None
            if ovectorizer is None:
                self.ovectorizer = vz.OneHotVectorizer()
                # print(y)
                self.y = self.ovectorizer.fit_transform(y)
            else:
                self.ovectorizer = ovectorizer
                self.y = self.ovectorizer.transform(y)
        else:
            if svectorizer is None:
                self.svectorizer = vz.SimpleVectorizer()
                self.y = self.ovectorizer.fit_transform(y)
            else:
                self.svectorizer = svectorizer
                self.y = self.svectorizer.transform(y)

    def __getitem__(self, i):
        X = self.X[i]
        if self.backwards: X = list(reversed(X))
        if self.eos is not None: X = X + [self.eos]
        if self.sos is not None: X = [self.sos]+ X
        return np.array(X), self.y[i]
    
    def __len__(self):
        return len(self.y)


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    def __len__(self):
        return len(self.y)


class SimpleBoWDataset(Dataset):
    def __init__(self, X, y, vocab_size):
        self.X = X
        self.y = y
        self.vocab_size = vocab_size
    
    def __getitem__(self, i):
        x = np.zeros(self.vocab_size)
        for j in self.X[i]: x[j] += 1
        return x, self.y[i]
    
    def __len__(self):
        return len(self.y)


class DuoDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = X1
        self.X2 = X2
        self.y = y
    
    def __getitem__(self, i):
        return self.X1[i], self.X2[i], self.y[i]
    
    def __len__(self):
        return len(self.y)

