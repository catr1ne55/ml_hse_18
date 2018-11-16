import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, confusion_matrix

data = pd.read_csv('classification.csv')
true_values = data[['true']]
pred_values = data[['pred']]

# tn, fp, fn, tp = confusion_matrix(true_values, pred_values).ravel()
# print(tp, fp, fn, tn)
#
# accuracy = accuracy_score(true_values,pred_values)
# precision = precision_score(true_values, pred_values)
# recall = recall_score(true_values, pred_values)
# f1 = f1_score(true_values, pred_values)
#
# print(accuracy, precision, recall, f1)
#
scores = pd.read_csv('scores.csv')
y_true = scores[['true']]
score_logreg = scores[['score_logreg']]
score_svm = scores[['score_svm']]
score_knn = scores[['score_knn']]
score_tree = scores[['score_tree']]
#
# logreg_auc = roc_auc_score(y_true, score_logreg)
# svm_auc = roc_auc_score(y_true, score_svm)
# knn_auc = roc_auc_score(y_true, score_knn)
# tree_auc = roc_auc_score(y_true, score_tree)
#
# print(logreg_auc, svm_auc, knn_auc, tree_auc)

l1, l2, l3 = precision_recall_curve(y_true, score_logreg)
logreg = list(zip(l1,l2,l3))
s1, s2, s3 = precision_recall_curve(y_true, score_svm)
svm = list(zip(s1, s2, s3))
k1, k2, k3 = precision_recall_curve(y_true, score_knn)
knn = list(zip(k1, k2, k3))
t1, t2, t3 = precision_recall_curve(y_true, score_tree)
tree = list(zip(t1, t2, t3))


def find_max_prec(clf):
    return max(list((map(lambda x: x[0] if x[1] > 0.7 else 0, clf))))

print(find_max_prec(logreg))
print(find_max_prec(svm))
print(find_max_prec(knn))
print(find_max_prec(tree))
