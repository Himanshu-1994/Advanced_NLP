import sys, os, csv, json, time, datetime, random, torch, argparse
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

from transformers import GPT2LMHeadModel
from transformers import BertLMHeadModel


from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)


def plotconf(mat,labels,save_dir):
    df_cm = pd.DataFrame(mat, index = [i for i in labels],columns=[i for i in labels])
    #df_cm = pd.DataFrame(mat, index = [i for i in ['anger','disgust','fear','joy','neutral','sadness','surprise']],
                  #columns = [i for i in ['anger','disgust','fear','joy','neutral','sadness','surprise']])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=False)
    plt.savefig(os.path.join(save_dir,'conf_mat'))
    return

def process_conf(flat_true_labels,flat_predictions):

    true_lab = []
    pred_lab = []
    for i in range(len(flat_true_labels)):
        l1 = np.argmax(flat_true_labels[i])
        l2 = np.argmax(flat_predictions[i])
        true_lab.append(l1)
        pred_lab.append(l2)
        """
        for j in range(len(flat_true_labels[i])):
            if flat_true_labels[i][j]==1:
                true_lab.append(j)
            if flat_predictions[i][j]==1:
                pred_lab.append(j)
        """
    return true_lab,pred_lab

def compute_metrics(labels_all, preds_all):
    assert len(preds_all) == len(labels_all)
    results = dict()

    labels = labels_all
    preds = preds_all
    #for i in range(len(labels_all)):
    #    preds = preds_all[i]
    #    labels = labels_all[i]
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

def predict(model, dataloader, args):
    print('Predicting labels for test sentences...')
    device = args.device
    model.eval() # Put model in evaluation mode

    predictions = [] # Tracking variable
    true_labels = [] # Tracking variable

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch) # Add batch to GPU

        b_input_ids, b_input_mask, b_labels = batch # Unpack the inputs from the dataloader


        with torch.no_grad(): # do not compute or store gradients to save memory and speed up prediction
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask) # Forward pass, calculate logit predictions
    
        #print("predict = ", outputs)

        logits = outputs[0] # retrieve the model outputs prior to activation

        logits = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))
        logits[logits>args.THRESHOLD] = 1
        logits[logits<=args.THRESHOLD] = 0

        #logits = logits.detach().cpu().numpy() # Move logits to CPU
        label_ids = b_labels.to('cpu').numpy() # Move labels to CPU
        
        predictions.append(logits)    # Store predictions
        true_labels.append(label_ids) # Store true labels

    print('COMPLETED.\n')
    return predictions, true_labels


def convert_to_one_hot_label(labs,classes):
    one_hot_label = [0] * classes
    for l in labs:
        one_hot_label[l] = 1
    return one_hot_label

def prepare_data(sentences, labels, tokenizer, args, random_sampling=False):

    input_ids=[]
    attention_masks = []
    labels = []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=args.MAX_LEN, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(encoded_dict['input_ids'])
    
    # Convert numpy arrays to pytorch tensors
    inputs = torch.cat(input_ids, dim=0)
    masks =  torch.cat(attention_masks, dim=0)
    labels = torch.cat(input_ids,dim=0)
    #labels = torch.tensor(labels,dtype=torch.float32)

    print("shape of labels = ", labels.shape)
    # Create the DataLoader for the given set (iterator to save memory)
    data = TensorDataset(inputs, masks, labels)
    if random_sampling: # train data
        sampler = RandomSampler(data)
    else: # dev and test data
        sampler = SequentialSampler(data)
    
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.BATCH_SIZE)
    return dataloader

def format_time(elapsed):
    """Function for formatting elapsed times: Takes a time in seconds and returns a string hh:mm:ss"""
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss

def train(model, tokenizer, optimizer, scheduler, train_dataloader, dev_dataloader,  args):
    total_t0 = time.time() # Measure the total training time for the whole run.
    training_stats = [] # training loss, validation loss, validation accuracy and timings.
    epochs = args.EPOCHS
    verbose = args.verbose
    device = args.device

    for epoch_i in range(0, epochs):
        if verbose: 
            print('\n======== Epoch {:} / {:} ========'.format(epoch_i+1, epochs))
        else: 
            print("Epoch",epoch_i+1,"of",epochs,"...")
        
        t0 = time.time()     # Measure how long the training epoch takes.
        total_train_loss = 0 # Reset the total loss for this epoch.
        model.train()        # Put the model into training mode.

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            
            if verbose==True and step%40==0 and not step==0: # Progress update every 40 batches.
                elapsed = format_time(time.time()-t0) # Calculate elapsed time in minutes.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader and copy each tensor to the GPU using the to() method.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            out = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = out.loss
            logits = out.logits
            #print("Loss = ", loss)
            #print("Logists shape = ", logits.shape)
            # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0 in order to prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step() # Update parameters and take a step using the computed gradient.
            scheduler.step() # Update the learning rate.

        avg_train_loss = total_train_loss / len(train_dataloader) # Calculate the average loss over the training data.
        if verbose: 
            print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
      
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        if verbose:
            print("  Training epoch took: {:}".format(training_time))


        # Record all statistics from this epoch.
        #avg_val_accuracy = 0
        #avg_val_loss = 0
        #validation_time = 0
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Train Loss': avg_train_loss,
                'Train Time': training_time,
            }
        )
      
        if args.save:
            # Save model checkpoint
            output_dir = os.path.join(args.save_dir, "checkpoint-{}".format(epoch_i+1))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            save_optimizer = True
            #torch.save(args, os.path.join(output_dir, "training_args.bin"))
            if save_optimizer:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    
    if args.save:
        statsfile = os.path.join(args.save_dir,"stats")
        with open(statsfile,"w") as f:
            for i,stat in enumerate(training_stats):
                f.write(json.dumps(stat))
                f.write("\n")

    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    #df_stats = pd.DataFrame(data=training_stats)
    #df_stats = df_stats.set_index('epoch')
    #return df_stats
    return training_stats



def run(args):

    models = {
        "ekman": ('bert-base-cased',"monologg/bert-base-cased-goemotions-ekman"),
        "group": ('bert-base-cased',"monologg/bert-base-cased-goemotions-group"),
        "original": ('bert-base-cased',"monologg/bert-base-cased-goemotions-original"),
        "multilingual": ('bert-base-multilingual-cased',"bert-base-multilingual-cased"),
        "bert": ('bert-base-cased',"bert-base-cased"),
        "gpt2": ('gpt2',"gpt2"),
        "finnish_cased": ('TurkuNLP/bert-base-finnish-cased-v1','TurkuNLP/bert-base-finnish-cased-v1'),
    }

    args.save_dir = os.path.join(args.save_dir,args.language+"-"+args.type)
    args.save_dir = os.path.join(args.save_dir,args.suff+"seed_"+str(args.SEED))
        
    print("Start training\n")

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"

    if args.use_pretrained:
        print("Using pre-trained model : ",args.pretrained_path)
        model_name_or_path = args.pretrained_path
        tokenizer_path = args.pretrained_path
    else:
        model_name_or_path = models[args.arch][0]
        tokenizer_path = models[args.arch][1]

    labels = []
    #label_file = "XED/AnnotatedData/labels_ekman.txt"
    label_file = "data/"+args.type+"/"+"labels.txt"

    with open(label_file,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            labels.append(line.rstrip())

    args.NUM_CLASSES = len(labels)

    model_config = BertConfig.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        )


    # Get model's tokenizer.
    print('Loading tokenizer...')

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
    )
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #tokenizer.pad_token = tokenizer.eos_token
    print("Load BERT Finnish model for LM training\n")

    model = BertLMHeadModel.from_pretrained(
       pretrained_model_name_or_path=model_name_or_path, 
       config=model_config,
       ignore_mismatched_sizes=True,
       )

    model.resize_token_embeddings(len(tokenizer))    
    model.config.pad_token_id = tokenizer.pad_token_id

    print("pad token id = ",model.config.pad_token_id)
    model.to(args.device)

    train_path = "data/"+args.type+"/lmtrain.tsv"

    if args.train_path != "":
        train_path = args.train_path

    print("train_path = ",train_path)

    df_train = pd.read_csv(train_path, delimiter='\t', header=None, names=['sentence','label']).sample(frac=1, random_state=args.SEED)

    X_train = df_train.sentence.values
    y_train = df_train.sentence.values


    if args.task=="train":
        
        train_dataloader      = prepare_data(X_train, y_train, tokenizer, args, True)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        adam         = AdamW(optimizer_grouped_parameters, lr=args.LEARN_RATE, eps=args.EPSILON)
        total_steps  = len(train_dataloader) * args.EPOCHS # Total number of training steps is nb_batches times nb_epochs.
        linear_sch   = get_linear_schedule_with_warmup(adam, num_warmup_steps=int(total_steps*args.nb_warmup_prop), num_training_steps=total_steps)

        training_stats = train(model=model, tokenizer = tokenizer, optimizer=adam, train_dataloader=train_dataloader, dev_dataloader=train_dataloader, 
                            scheduler=linear_sch, args= args)
        print(training_stats)
   
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
    
def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type", type=str, default="original")
    parser.add_argument("--task", type=str, default="train")
    parser.add_argument("--arch", type=str, default="bert")
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--gen_labels", type=bool, default=False)
    parser.add_argument("--demoji", type=bool, default=False)

    parser.add_argument("--save_dir", type=str, default="/nobackup3/wb/xed/")
    parser.add_argument("--suff", type=str, default="")    
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--dev_path", type=str, default="")
    parser.add_argument("--label_file", type=str, default="")


    parser.add_argument("--SEED", type=int, default=100)
    parser.add_argument("--EPOCHS", type=int, default=10)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--language", type=str, default="goemotions")
    parser.add_argument("--checkpoint",  type=int, default=10)

    parser.add_argument("--use_pretrained",  type=bool, default=False)
    parser.add_argument("--pretrained_path",  type=str, default="")

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
    run(args)
