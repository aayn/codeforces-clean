from collections import Counter
import numpy as np
from torch import nn
from sklearn.metrics import precision_recall_fscore_support


def mapt(f, *iters):
    return tuple(map(f, *iters))

def mapl(f, *iters):
    return list(map(f, *iters))

def get_classification_type(y):
    if all(map(lambda x: type(x) is str or len(x) == 1, y)):
        return 'multiclass'
    return 'multilabel'

def multi_to_text_out(outs, vectorizer):
    itos = vectorizer.itos
    if len(outs[0]) > 1:
        outs = np.argmax(outs, axis=1)
    outs = mapl(lambda x: itos[x], outs)
    return outs

def weights_init(m):
    try:
        nn.init.xavier_uniform_(m.weight.data)
    except (AttributeError, ValueError):
        pass

def multilabel_prediction(preds, threshold=0.2):
    return np.array(preds>threshold, dtype=np.int)


def vote_pred(preds):
    print(preds)
    return mapl(lambda *x: Counter(tuple(x)).most_common(1)[0][0], *preds)


def prf_report(y_targs, y_preds, labels):
    p, r, f, s = precision_recall_fscore_support(y_targs, y_preds)
    print('Label\tPrecsion\tRecall\tFScore\tSupport')
    for i, label in enumerate(labels):
        print(f'{label}\t{p[i]}\t{r[i]}\t{f[i]}\t{s[i]}')