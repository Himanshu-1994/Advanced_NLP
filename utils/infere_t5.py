import pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score



def compute_metrics(labels_all, preds_all):
    assert len(preds_all) == len(labels_all)
    results = dict()

    labels = labels_all
    preds = preds_all
    try:
        results["accuracy"] = accuracy_score(labels, preds)
        results["macro_precision"], results["macro_recall"], results[
            "macro_f1"], _ = precision_recall_fscore_support(
            labels, preds, average="macro")
        results["micro_precision"], results["micro_recall"], results[
            "micro_f1"], _ = precision_recall_fscore_support(
            labels, preds, average="micro")
        results["weighted_precision"], results["weighted_recall"], results[
            "weighted_f1"], _ = precision_recall_fscore_support(
            labels, preds, average="weighted")

    except:
        print("Erorr at index = ", i,"\n")
        print("Pred =", preds)
        print("lables = ", labels)

    return results

def convert_to_one_hot_label(labs,num_classes):
    one_hot_label = [0] * num_classes
    for l in labs:
        one_hot_label[l] = 1
    return one_hot_label


def func(x,labeltoid):
    names = x.split(',')
    labs = [labeltoid[name] if name in labeltoid else labeltoid['neutral'] for name in names]
    return labs


truelabels = []
label_file = "data/original/labels.txt"

with open(label_file,"r",encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        truelabels.append(line.rstrip())

labeltoid = {}

for i,label in enumerate(truelabels):
    labeltoid[label] = i

path = "/nobackup3/himanshu/xed/goemotions-original/t5result/pred_res"
with open(path,'rb') as f:
    res = pickle.load(f)

truth = res['truth']
truth = [func(x,labeltoid) for x in truth]
truth = [convert_to_one_hot_label(l,len(truelabels)) for l in truth]

preds = res['preds']

preds = [func(x,labeltoid) for x in preds]
preds = [convert_to_one_hot_label(l,len(truelabels)) for l in preds]
output = compute_metrics(truth, preds)
print(output)
 