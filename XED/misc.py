import sys, io, os, re, csv, json, string, time, datetime, random, unicodedata, itertools, collections, torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils import resample
from model_xed import BertForMultiLabelClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def plot_lengths_distribution(lengths, title="Length distribution of tokenized sentences"):
  sns.set(style="darkgrid")
  sns.set(font_scale=1.5)
  plt.rcParams["figure.figsize"]=(10,5)
  lengths = [min(length,MAX_LEN) for length in lengths]
  ax = sns.distplot(lengths,kde=False,rug=False, hist_kws={"rwidth":5,'edgecolor':'black', 'alpha':1.0}) 
  plt.title("Sequence length distribution")
  plt.xlabel("Sequence Length")
  plt.ylabel("Counts")

  num_truncated = lengths.count(MAX_LEN)
  num_sentences = len(lengths)
  print("{:.1%} of the training examples ({:,} of them) have more than {:,} tokens".format(float(num_truncated)/float(num_sentences),num_truncated,MAX_LEN))

def plot_value_counts(df, title="Label distribution"):
  emotions = ['TRUST','ANGRY','ANTICIP.','DISGUST','FEAR','JOY','SADNESS','SURPRISE']
  df2 = df.replace([0,1,2,3,4,5,6,7], emotions)
  df2.label.value_counts(normalize=False).sort_index().plot(kind='bar')
  plt.xticks(rotation=25)
  plt.title(title)
  plt.show()
  plt.savefig('saved classes')

def plot_loss(df_stats, plot_train=True, plot_valid=True):
  sns.set(style='darkgrid') # Use plot styling from seaborn.
  sns.set(font_scale=1.5) # Increase the plot size.
  plt.rcParams["figure.figsize"] = (12,6) # Increase the font size.
  if plot_train: plt.plot(df_stats['Train Loss'], 'b-o', label="Training")
  if plot_valid: plt.plot(df_stats['Valid Loss'], 'g-o', label="Validation")
  plt.title("Training & Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.xticks([1, 2, 3, 4])
  plt.show()

def generate_pseudo_data(predictions,true_labels,output_file,X_all):
    print("Writing pseudo labels to output\n")
    output_file_true = output_file.split(".")[0]+"-truelabel.tsv"
    pred_lab_data = []
    true_lab_data = []

    for t, label in enumerate(predictions):
        labs = []
        for index in range(len(label)):
            if label[index]==1:
                labs.append(index)

        labstrn = ",".join(str(i) for i in labs)
    
        if labstrn is None or len(labstrn)==0:
            continue
    
        text = X_all[t]
        pred_lab_data.append(text+"\t"+labstrn+"\n")
        true_lab_data.append(text+"\t"+true_labels[t]+"\n")

    with open(output_file,"w", encoding="utf-8") as f:
        for line in pred_lab_data:
            f.write(line)

    with open(output_file_true,"w", encoding="utf-8") as f:
        for line in true_lab_data:
            f.write(line)


def evaluate(predictions, true_labels, avg='macro', verbose=True):
  avgs = ['micro', 'macro', 'weighted', 'samples']
  if avg not in avgs:
    raise ValueError("Invalid average type (avg). Expected one of: %s" % avgs)

  # Combine the predictions for each batch into a single list.
  flat_predictions = [item for sublist in predictions for item in sublist]
  #flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

  # Combine the correct labels for each batch into a single list.
  flat_true_labels = [item for sublist in true_labels for item in sublist]

  # Compute the results.
  precision = precision_score(flat_true_labels, flat_predictions, average=avg)
  recall    = recall_score(flat_true_labels, flat_predictions, average=avg)
  f1        = f1_score(flat_true_labels, flat_predictions, average=avg)
  acc       = accuracy_score(flat_true_labels,flat_predictions)

  # Report the results.
  if verbose:
    print('Accuracy:        %.4f' % acc)
    print(avg+' Precision: %.4f' % f1)
    print(avg+' Recall:    %.4f' % f1)
    print(avg+' F1 score:  %.4f' % f1, "\n")
    #print(confusion_matrix(flat_true_labels,flat_predictions))
    #print(classification_report(flat_true_labels, flat_predictions, digits=2, zero_division='warn'))

  return f1, acc



  
def flat_accuracy(preds, labels):
    """Function to calculate the accuracy of our predictions vs labels"""
    #preds = 1 / (1 + np.exp(-preds.detach().cpu().numpy()))
    #preds[preds > 0.3] = 1

    #pred_flat = np.argmax(preds, axis=1).flatten()
    #labels_flat = labels.flatten()

    pred_flat = np.argmax(preds, axis=1)
    #labels_flat = labels

    correct = 0
    for i in range(len(pred_flat)):
        pred = pred_flat[i]
        labs = labels[i]
        if labs[pred] == 1:
            correct+=1

    return correct/len(labels)

    #return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    """Function for formatting elapsed times: Takes a time in seconds and returns a string hh:mm:ss"""
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss

def print_model_params(my_model):
  """Function to print all the model's parameters as a list of tuples: (name,dimensions)"""
  params = list(my_model.named_parameters())
  print('The BERT model has {:} different named parameters.\n'.format(len(params)))
  print('==== Embedding Layer ====\n')
  for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
  print('\n==== First Transformer ====\n')
  for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
  print('\n==== Output Layer ====\n')
  for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))



def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    results = dict()

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

    return results
