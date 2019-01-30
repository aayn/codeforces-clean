import pickle
import random
from functools import reduce, partial
import dill
import numpy as np
from pathlib import Path

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import Subset, ConcatDataset

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import hamming_loss, f1_score, classification_report

from skai.dataset import TokenDataset, SimpleDataset
from skai.sanitizer import sample_cleaner, small_class, small_class_multi
from skai.utils import get_classification_type, weights_init, multilabel_prediction
from skai.vectorizer import OneHotVectorizer
from skai.metrics import hamming_loss_skai

SKAI_COMMON = Path('skai/common/')


class TextRunner:
    def __init__(self, mwrappers, rdata, labels, dataset_name,
                 make_pyt_data=True, verbose=True):
        """Takes different mwrappers and runs training and evaluation
        over them.
        
        mwrappers: A list of mwrappers.
        rdata: Raw data, list of strings.
        labels: Labels, list of strings.
        """
        self.verbose = verbose
        dill.settings['recurse'] = True
        try:
            rdata, labels = pickle.load((SKAI_COMMON/f'{dataset_name}.pkl').open('rb'))
        except FileNotFoundError:
            # rdata, labels = sample_cleaner([small_class], rdata, labels)
            pickle.dump((rdata, labels), (SKAI_COMMON/f'{dataset_name}.pkl').open('wb'))
        if verbose:
            print('Checkpoint reached: raw data cleaned.')
        
        self.rdata = rdata
        self.labels = labels

        self.classification_type = get_classification_type(labels)
        if verbose:
            print(f'{self.classification_type} classification.')

        try:
            self.alldata = dill.load((SKAI_COMMON/f'{dataset_name}_alldata.pkl').open('rb'))
        except (FileNotFoundError, EOFError):
            self.alldata = TokenDataset(rdata, labels, onehot=True)
            dill.dump(self.alldata, (SKAI_COMMON/f'{dataset_name}_alldata.pkl').open('wb'))

        self.mwrappers = mwrappers
        if make_pyt_data:
            self.data_setup_pyt(dataset_name)

    def data_setup_pyt(self, dataset_name):
        self.trainsets_pyt = []
        self.valsets_pyt = []

        trainset = self.get_dataset(f'{dataset_name}_trainset0', None, None,
                                        TokenDataset, onehot=True,
                                        tvectorizer=self.alldata.tvectorizer,
                                        ovectorizer=self.alldata.ovectorizer)
        valset = self.get_dataset(f'{dataset_name}_valset0', None, None,
                                      TokenDataset, onehot=True,
                                      tvectorizer=self.alldata.tvectorizer,
                                      ovectorizer=self.alldata.ovectorizer)
        testset = self.get_dataset(f'{dataset_name}_testset4', None, None,
                                   TokenDataset, onehot=True,
                                   tvectorizer=self.alldata.tvectorizer,
                                   ovectorizer=self.alldata.ovectorizer)
        
        Xall, yall = [], []
        allset = ConcatDataset([trainset, valset, testset])
        for x, y in allset:
            Xall.append(x)
            yall.append(y)
        
        self.dataset = [np.array(Xall), np.array(yall)]
    
    def get_clf_sk(self, mwrapper, X, y, scoring='accuracy'):
        pipeline = mwrapper.pipeline
        parameters = mwrapper.parameters
        if self.classification_type == 'multilabel':
            # ohv = OneHotVectorizer()
            print(set(reduce(lambda acc, x: acc + x, y, [])))
            print(len(set(reduce(lambda acc, x: acc + x, y, []))))
            # print(y[1])
            y = self.alldata.ovectorizer.transform(y)
            # y = ohv.fit_transform(y)
            # print(y[1])
        grid_search_tune = GridSearchCV(pipeline, parameters,
                                        n_jobs=4, verbose=1, scoring=scoring)
        grid_search_tune.fit(X, y)
        return grid_search_tune.best_estimator_, grid_search_tune.best_score_
        

    def get_learner_pyt(self, mwrapper, trainset, valset, ce_loss=False):

        dl_train = fd.DataLoader(trainset, batch_size=32,
                                 num_workers=1, pad_idx=1,
                                 transpose=False)
        dl_val = fd.DataLoader(valset, batch_size=32,
                               num_workers=1, pad_idx=1,
                               transpose=False)
        # dl_test = fd.DataLoader(self.testset, batch_size=4,
        #                         num_workers=2, pad_idx=1,
        #                         transpose=False)
        modeldata = fs.ModelData(f'skai/models/{mwrapper.name}', dl_train, dl_val)
        # print(modeldata.trn_y)
        mwrapper.model.apply(weights_init)
        crit = None
        if ce_loss:
            w = torch.Tensor([0.08148148148148147, 0.22916666666666666,
                              0.25, 0.2727272727272727, 0.3, 0.5076923076923077,
                              0.515625, 0.673469387755102, 0.9166666666666666, 1.0])
            crit = CE_lambda(w)
        learner = fl.Learner.from_model_data(mwrapper.model, modeldata,
                                             opt_fn=Adam_lambda(), crit=crit)
        if self.classification_type == 'multilabel':
            learner.metrics = [fm.accuracy_multi, hamming_loss_skai]
        else:
            learner.metrics = [fm.accuracy, fm.recall]
        return learner
    
    def load_learner_pyt(self, mwrapper, trainset, valset, ce_loss=False):
        learner = self.get_learner_pyt(mwrapper, trainset, valset, ce_loss)
        learner.load('best')
        return learner        

    def vis_lr_pyt(self, learner):
        learner.sched.plot()

    def fit_pyt(self, learner, lrs, epochs):
        learner.fit(lrs, epochs)
    
    def get_dataset(self, name, X, y, Dataset=TokenDataset, **kwargs):
        try:
            # raise FileNotFoundError
            dataset = dill.load((SKAI_COMMON/f'{name}.pkl').open('rb'))
        except (FileNotFoundError, EOFError):
            dataset = Dataset(X, y, **kwargs)
            dill.dump(dataset, (SKAI_COMMON/f'{name}.pkl').open('wb'))
        return dataset

    
    def run(self, lrs=1e-4, epochs=5):
        for mwrapper in self.mwrappers:
            if mwrapper.type == 'sklearn':
                clf = self.get_clf_sk(mwrapper)
                predictions = clf.predict(self.X_test)
                print(classification_report(self.y_test, predictions))
            else:
                best_acc = 0
                best_f1_mic = 0
                for trs, vals in zip(self.trainsets_pyt, self.valsets_pyt):
                    # Set True when use cross entropy loss
                    # print(trs[0])
                    learner = self.get_learner_pyt(mwrapper, trs, vals, False)
                    self.fit_pyt(learner, lrs, epochs)
                    dl_test = fd.DataLoader(self.testset, batch_size=32,
                                            num_workers=2, pad_idx=1,
                                            transpose=False)
                    preds, targs = learner.predict_dl(dl_test)
                    if self.verbose:
                        print(targs[:1], preds[:1])
                    if self.classification_type == 'multilabel':
                        preds = multilabel_prediction(preds, threshold=0.5)
                        # hl = hamming_loss(targs, preds)
                        micro_f1 = f1_score(targs, preds, average='micro')
                        # macro_f1 = f1_score(targs, preds, average='macro')
                        print(micro_f1)
                        if micro_f1 > best_f1_mic:
                            best_f1_mic = micro_f1
                            learner.save(f'best')

                    else:
                        acc = float(fm.accuracy(torch.from_numpy(preds), torch.from_numpy(targs)))
                        print(acc)
                        if acc > best_acc:
                            best_acc = acc
                            learner.save('best')
                    # print(fm.precision(torch.from_numpy(preds), torch.from_numpy(targs)))
                    # print(fm.recall(torch.from_numpy(preds), torch.from_numpy(targs)))
                    # print(fm.f1(torch.from_numpy(preds), torch.from_numpy(targs)))
                return (preds, targs)
    
    def loaded_run(self, model='test'):
        mwrapper = self.mwrappers[0]
        for trs, vals in zip(self.trainsets_pyt, self.valsets_pyt):
            learner = self.get_learner_pyt(mwrapper, trs, vals, False)
        learner.load('best')
        
        testset = self.testset
        if model == 'val':
            testset = self.valsets_pyt[0]

        dl_test = fd.DataLoader(testset, batch_size=32,
                                num_workers=2, pad_idx=1,
                                transpose=False)
        preds, targs = learner.predict_dl(dl_test)
        return (preds, targs)

        
                

def Adam_lambda(lr=0.001):
    return lambda *args, **kwargs: Adam(*args, lr=lr, *kwargs)

def SGD_Momentum(momentum):
    return lambda *args, **kwargs: optim.SGD(*args, momentum=momentum, **kwargs)

def CE_lambda(w):
    crit = partial(F.cross_entropy, weight=fc.to_gpu(w))
    def loss(y_pred, y_true):
        y_true = torch.argmax(y_true, -1)
        return crit(y_pred, y_true)
    return loss