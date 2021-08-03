import torch
import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
from ._utils import *

def get_metric(is_multiclass):
    if is_multiclass:
        metric = DefaultClassificationMetrics
    else:
        metric = MultiLabelClassificationMetrics
    return metric

def get_activation(is_multiclass):
    if is_multiclass:
        act = nn.Softmax(dim=1)
    else:
        act = nn.Sigmoid()
    return act

def mean_roc_auc(truths, predictions):
    """
    Calculating mean ROC-AUC:
        Asuuming that the last dimension represent the classes
    """
    _truths = np.array(deepcopy(truths))
    _predictions = np.array(deepcopy(predictions))  
    n_classes = _predictions.shape[-1]
    avg_roc_auc = 0 
    for class_num in range(n_classes):
        auc = 0.5
        tar = (_truths[:,class_num] + _truths[:,class_num]**2 ) / 2
        if tar.sum() > 0:
            auc = metrics.roc_auc_score(tar, _predictions[:,class_num], 
                                        average='macro', 
                                        sample_weight=_truths[:, class_num] ** 2 + 1e-06, 
                                        multi_class = 'ovo')            
        avg_roc_auc += auc 
    return avg_roc_auc / n_classes

class DefaultClassificationMetrics:
    
    def __init__(self, n_classes, int_to_labels=None, act_threshold=0.5, mode=""):
        self.mode = mode
        self.prefix = ""
        if mode:
            self.prefix = mode + "_"
        self.n_classes = n_classes
        if int_to_labels is None:
            int_to_labels = {val:'class_'+str(val) for val in range(n_classes)}
        self.int_to_labels = int_to_labels
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.truths = []
        self.predictions = []
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.truths = []
        self.predictions = []        
    
    # add predictions to confusion matrix etc
    def add_preds(self, y_pred, y_true, using_knn=False):
        if not using_knn:
            y_pred = y_pred.max(dim = 1)[1].data
        y_true = y_true.flatten().detach().cpu().numpy()
        y_pred = y_pred.flatten().detach().cpu().numpy()
        self.truths += (y_true.tolist())
        self.predictions += (y_pred.tolist())
        np.add.at(self.confusion_matrix, (y_true, y_pred), 1)
    
    # Calculate and report metrics
    def get_value(self, use_dist=True):
        if use_dist:
            synchronize()
            truths = sum(dist_gather(self.truths), [])
            predictions = sum(dist_gather(self.predictions), [])
        else:
            truths = self.truths
            predictions = self.predictions     
        
        accuracy = metrics.accuracy_score(truths, predictions)
        precision = metrics.precision_score(truths, predictions, average='macro', zero_division=0)
        recall = metrics.recall_score(truths, predictions, average='macro', zero_division=0)
        f1 = metrics.f1_score(truths, predictions, average='macro', zero_division=0) 
        kappa = metrics.cohen_kappa_score(truths, predictions, 
                                          labels=list(range(self.n_classes)), weights='quadratic')
        
        # return metrics as dictionary
        return edict({self.prefix + "accuracy" : round(accuracy, 3),
                        self.prefix + "precision" : round(precision, 3),
                        self.prefix + "recall" : round(recall, 3),
                        self.prefix + "f1" : round(f1, 3),
                        self.prefix + "cohen_kappa" : round(kappa, 3)})
    
class MultiLabelClassificationMetrics:
    
    def __init__(self, n_classes, int_to_labels=None, act_threshold=0.5, mode=""):
        self.mode = mode
        self.prefix = ""
        if mode:
            self.prefix = mode + "_"       
        self.n_classes = n_classes
        if int_to_labels is None:
            int_to_labels = {val:'class_'+str(val) for val in range(n_classes)}
        self.labels = np.arange(n_classes)
        self.int_to_labels = int_to_labels
        self.truths = []
        self.predictions = []
        self.activation = nn.Sigmoid()
        self.act_threshold = act_threshold
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.truths = []
        self.predictions = []        
    
    # add predictions to confusion matrix etc
    def add_preds(self, y_pred, y_true, using_knn=False):       
        y_true = y_true.int().detach().cpu().numpy()
        y_pred = self.preds_from_logits(y_pred)
        self.truths += (y_true.tolist())
        self.predictions += (y_pred.tolist())
    
    # pass signal through activation and thresholding
    def preds_from_logits(self, preds):
        preds = self.activation(preds)
        return preds.detach().cpu().numpy()
    
    def threshold_preds(self, preds):
        preds = preds > self.act_threshold
        if isinstance(preds, torch.Tensor):
            return preds.int().detach().cpu().numpy()
        else:
            return preds * 1
    
    # Calculate and report metrics
    def get_value(self, use_dist=True):
        if use_dist:
            synchronize()
            truths = np.array(sum(dist_gather(self.truths), []))
            predictions = np.array(sum(dist_gather(self.predictions), []))
        else:
            truths = np.array(self.truths)
            predictions = np.array(self.predictions) 
            
        try:
            mAP = metrics.average_precision_score(truths, predictions, average='macro')
        except:
            mAP = 0.                    
        roc_auc = mean_roc_auc(truths, predictions)        
        
        predictions = self.threshold_preds(predictions)
        self.confusion_matrix = metrics.multilabel_confusion_matrix(truths, predictions)     
        
        accuracy = metrics.accuracy_score(truths, predictions)
        precision = metrics.precision_score(truths, predictions, average='macro', 
                                            labels=self.labels, zero_division=0)
        recall = metrics.recall_score(truths, predictions, average='macro', 
                                            labels=self.labels, zero_division=0)
        f1 = metrics.f1_score(truths, predictions, average='macro', 
                                            labels=self.labels, zero_division=0)
        
        # return metrics as dictionary
        return edict({self.prefix + "accuracy" : round(accuracy, 3),
                        self.prefix + "mAP" : round(mAP, 3),
                        self.prefix + "precision" : round(precision, 3),
                        self.prefix + "recall" : round(recall, 3),
                        self.prefix + "f1" : round(f1, 3),
                        self.prefix + "roc_auc" : round(roc_auc, 3)}
                    )    
    