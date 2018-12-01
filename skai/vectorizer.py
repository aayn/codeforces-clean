from abc import ABC, abstractmethod
from collections import Counter
from functools import reduce
import numpy as np
from fastai import text as ft
from fastai import dataloader as fd
from fastai import dataset as fs
from fastai import learner as fl
from fastai import core as fc
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from skai.utils import get_classification_type


class Vectorizer(ABC):
    @abstractmethod
    def fit(self, data):
        "data is a list of str."
    
    @abstractmethod
    def transform(self, data):
        "data is a list of str."
    
    def fit_transform(self, data):
        "data is a list of str."
        self.fit(data)
        return self.transform(data)


class TokenVectorizer(Vectorizer):
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.itos, self.stoi, self.tokens = None, None, None
    
    def fit(self, data):
        "data: (list of str)."
        tokens = ft.Tokenizer().proc_all_mp(fc.partition_by_cores(data))
        self.itos, self.stoi = ft.numericalize_tok(tokens, min_freq=self.min_freq)
    
    def transform(self, data):
        assert(self.itos is not None)
        assert(self.stoi is not None)
        self.tokens = [[tok for tok in ft.Tokenizer().proc_text(sent)]
                          for sent in data]
        self.tok_vecs = [[self.stoi[tok] for tok in ft.Tokenizer().proc_text(sent)]
                          for sent in data]
        return self.tok_vecs, self.tokens



class SimpleVectorizer(Vectorizer):
    def __init__(self):
        self.itos, self.stoi = None, None
        self.classification_type = 'multiclass'
    
    def fit(self, data):
        self.classification_type = get_classification_type(data)
        if self.classification_type == 'multilabel':
            data = reduce(lambda acc, x: acc + list(x), data, [])
        self.itos = [i for i, _ in Counter(data).most_common()]
        self.stoi = {l: i for i, l in enumerate(self.itos)}
    
    def transform(self, data):
        if self.classification_type == 'multiclass':
            return np.array(tuple(map(lambda x: [float(self.stoi[x])], data)))
        # print(data)
        return np.array([tuple([int(self.stoi[cl]) for cl in classes])
                        for classes in data])
        


class OneHotVectorizer(SimpleVectorizer):
    def __init__(self):
        self.itos, self.stoi, self.ohe = None, None, None

    def fit(self, data):
        self.mlb = MultiLabelBinarizer()
        super().fit(data)
        nums = super().transform(data)
        # print(nums)
        self.mlb.fit(list(nums))
    
    def transform(self, data):
        nums = super().transform(data)
        return self.mlb.transform(nums)

