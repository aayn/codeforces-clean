import numpy as np

from sklearn.metrics import hamming_loss, f1_score

from skai.utils import multilabel_prediction


def hamming_loss_skai(y_preds, y_targs):
    preds = multilabel_prediction(y_preds, 0.5)
    return hamming_loss(y_targs, preds)

def f1_micro_skai(y_preds, y_targs):
    preds = multilabel_prediction(y_preds, 0.5)
    return f1_score(y_targs, preds, average='micro')