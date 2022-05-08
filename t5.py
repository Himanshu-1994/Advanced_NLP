import sys, os, csv, json, time, datetime, random, torch, argparse, glob

#cache_dir = 'tmp'
#cache_dir = '/nobackup3/himanshu/cache/'
#os.environ['TRANSFORMERS_CACHE'] = cache_dir

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils import resample
#from model import BertClassifier
from attrdict import AttrDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix
from simpletransformers.t5 import T5Model


from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)



truelabels = []
label_file = "data/original/labels.txt"

with open(label_file,"r",encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        truelabels.append(line.rstrip())

labeltoid = {}
for i,label in enumerate(truelabels):
    labeltoid[label] = i


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

def convert_to_one_hot_label(labs,classes):
    one_hot_label = [0] * classes
    for l in labs:
        one_hot_label[l] = 1
    return one_hot_label

def func(x):
    
    labs = list(map(int, x.split(",")))
    labnames = [truelabels[x] for x in labs]
    return ",".join(labnames)

def converttoid(x):
    names = x.split(',')
    labs = [labeltoid[name] for name in names if name in labeltoid]
    #labs = [labeltoid[name] if name in labeltoid else labeltoid['neutral'] for name in names]
    return labs

def inference(res):
    truth = res['truth']
    truth = [converttoid(x) for x in truth]
    truth = [convert_to_one_hot_label(l,len(truelabels)) for l in truth]

    preds = res['preds']

    preds = [converttoid(x) for x in preds]
    preds = [convert_to_one_hot_label(l,len(truelabels)) for l in preds]
    output = compute_metrics(truth, preds)
    
    report = classification_report(truth,preds,target_names=truelabels)

    return output,report


def run(args):

    args.save_dir = os.path.join(args.save_dir,args.language+"-"+args.type)
    args.save_dir = os.path.join(args.save_dir,args.suff+"seed_"+str(args.SEED))
        
    labels = truelabels
    args.NUM_CLASSES = len(labels)

    print("Start training\n")

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"

    if args.use_pretrained:

        if args.pretrained_path == "":
            ckpt = args.checkpoint
            folder = glob.glob(args.save_dir+"/checkpoint-*-"+str(ckpt))[0]
            args.pretrained_path = os.path.join(args.save_dir, folder)
        
        model_name_or_path = args.pretrained_path
        
    else:
        model_name_or_path = "t5-base"
        tokenizer_path = "t5-base"

    print("Load T5 model\n")


    model_args = {
        "max_seq_length": 50,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "num_train_epochs": args.EPOCHS,
        "evaluate_during_training": False,
        "evaluate_during_training_steps": 3000,
        "evaluate_during_training_verbose": False,
        "use_multiprocessing": False,
        "fp16": False,
        "save_eval_checkpoints": True,
        "save_model_every_epoch": True,
        "save_steps": -1,
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "special_tokens_list":['[RELIGION]','[NAME]'],
        "cache_dir": cache_dir,
        'output_dir':args.save_dir,
        "use_multiprocessing_for_evaluation": False,
        "process_count": 1,
    }


    #train_path = "data/"+args.type+"/augmented/train_aug_onlysamplelowclasses.tsv"
    train_path = "data/"+args.type+"/train.tsv"
    dev_path = "data/"+args.type+"/dev.tsv"
    test_path = "data/"+args.type+"/test.tsv"

    if args.train_path != "":
        train_path = args.train_path

    if args.dev_path != "":
        dev_path = args.dev_path

    if args.test_path != "":
        test_path = args.test_path

    print("train_path = ",train_path)

    df_train = pd.read_csv(train_path, delimiter='\t', header=None, names=['input_text','target_text','extra']).sample(frac=1, random_state=args.SEED)
    df_dev = pd.read_csv(dev_path, delimiter='\t', header=None, names=['input_text','target_text','extra']).sample(frac=1, random_state=args.SEED)
    df_test = pd.read_csv(test_path, delimiter='\t', header=None, names=['input_text','target_text','extra']).sample(frac=1, random_state=args.SEED)

    df_train = df_train[['input_text','target_text']].astype(str)
    df_dev = df_dev[['input_text','target_text']].astype(str)
    df_test = df_test[['input_text','target_text']].astype(str)

    df_train['target_text'] = df_train['target_text'].apply(func)
    df_dev['target_text'] = df_dev['target_text'].apply(func)
    df_test['target_text'] = df_test['target_text'].apply(func)

    df_train["prefix"] = "emotion"
    df_dev["prefix"] = "emotion"
    df_test["prefix"] = "emotion"


    # Prepare the data for testing
    to_predict = [
        prefix + ": " + str(input_text)
        for prefix, input_text in zip(df_test["prefix"].tolist(), df_test["input_text"].tolist())
    ]
    truth = df_test["target_text"].tolist()

    #model_path = "t5-base"
    #model_path = "/nobackup3/himanshu/xed/goemotions-original/t5result/paraphrase"+args.suff
    #"checkpoint-13570-epoch-5"
    model = T5Model("t5", model_name_or_path, args=model_args)

    result_dict = {}
    result_dict['truth'] = []
    result_dict['preds'] = []

    if args.task=="train":
        model.train_model(df_train, output_dir=args.save_dir)        
        #model.train_model(df_train, eval_data=df_dev,output_dir="/nobackup3/himanshu/xed/goemotions-original/t5result/")        
        preds = model.predict(to_predict)
        print("\n\nPredicting done on Test split\n")

        for truth_value, pred in zip(truth, preds):
            result_dict["truth"].append(truth_value)
            result_dict["preds"].append(pred)

        import pickle
        with open(os.path.join(args.save_dir,"pred_res"),'wb') as f:
            pickle.dump(result_dict,f)

    elif args.task == "test":
        
        preds = model.predict(to_predict)
        for truth_value, pred in zip(truth, preds):
            result_dict["truth"].append(truth_value)
            result_dict["preds"].append(pred)

        results,report = inference(result_dict)
        print(results)

        save_path = args.pretrained_path
        resultfile = os.path.join(save_path,"results_test")
        print(report)
        with open(resultfile,"w") as f:
            f.write(report)
            f.write("\n")


def set_seeds(seed=15):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type", type=str, default="original")
    parser.add_argument("--task", type=str, default="train")
    parser.add_argument("--arch", type=str, default="t5-base")

    parser.add_argument("--save_dir", type=str, default="/nobackup3/wb/xed/")
    parser.add_argument("--suff", type=str, default="")
    
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--dev_path", type=str, default="")


    parser.add_argument("--SEED", type=int, default=10)
    parser.add_argument("--EPOCHS", type=int, default=10)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--language", type=str, default="goemotions")

    parser.add_argument("--use_pretrained",  type=bool, default=False)
    parser.add_argument("--pretrained_path",  type=str, default="")
    parser.add_argument("--checkpoint",  type=int, default=10)

    parser.add_argument("--THRESHOLD",  type=int, default=0.3)
    parser.add_argument("--BATCH_SIZE",  type=int, default=16)
    parser.add_argument("--MAX_LEN",  type=int, default=50)
    parser.add_argument("--nb_warmup_prop",  type=float, default=0.1)
    parser.add_argument("--LEARN_RATE", type=float, default=5e-5)
    parser.add_argument("--EPSILON",  type=float, default=1e-8)
    parser.add_argument("--weight_decay",  type=float, default=0.0)

    parser.add_argument("--NUM_CLASSES",  type=int, default=7)
    parser.add_argument("--save",  type=bool, default=True)
    parser.add_argument("--verbose",  type=bool, default=True)


    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    set_seeds(args.SEED)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    run(args)
