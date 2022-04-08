import sys, io, os, re, csv, json, string, time, datetime, random, unicodedata, itertools, collections, torch, argparse
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

from misc import *

def cross_validation(args, tokenizer):
    
    df_dataset = pd.read_csv(args.data_path, delimiter='\t', header=None, names=['sentence','label']).sample(frac=1, random_state=args.SEED)

    print("Performing %d-fold cross-validation,".format(NUM_FOLDS))
    kf = KFold(n_splits=args.NUM_FOLDS, random_state=args.SEED, shuffle=True)
    mf1s = []
    accs = []
    fold_num = 1
    
    for train_index, test_index in kf.split(df_dataset):
        print("##### Fold number:", fold_num, "#####")
        fold_num += 1
        train_df = df_dataset.iloc[train_index]
        test_df  = df_dataset.iloc[test_index]

        train_df, dev_df = train_test_split(train_df, test_size=0.125) # change percentage (hyperparams?)

        X_train, y_train = train_df.sentence.values, train_df.label.values
        X_dev, y_dev = dev_df.sentence.values, dev_df.label.values
        X_test, y_test = test_df.sentence.values, test_df.label.values

        train_dataloader      = prepare_data(X_train, y_train, tokenizer, args, 1)
        dev_dataloader        = prepare_data(X_dev, y_dev, tokenizer, args, 0)
        prediction_dataloader = prepare_data(X_test, y_test, tokenizer, args, 0)

        training_stats = train(model, tokenizer = tokenizer, optimizer=adam, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, epochs=EPOCHS, scheduler=linear_sch)
        predictions, true_labels = predict(model, prediction_dataloader, args)
        mf1, acc = evaluate(predictions, true_labels, verbose=True)
        mf1s.append(mf1)
        accs.append(acc)

    print("#####################################################################")
    print("PARAMS: epochs:",EPOCHS,", lr_rate:",LEARN_RATE,"epsilon:",EPSILON,"...")
    print("#####################################################################")
    print("F1(CV):",mf1s)
    print(f"Mean-folds-F1: {sum(mf1s)/len(mf1s)}")
    print("Acc(CV):",accs)
    print(f"Mean-folds-Acc: {sum(accs)/len(accs)}")

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
    #labels = [int(s.split(",")[0]) for s in labels]
    labels = [list(map(int,s.split(","))) for s in labels]
    labels = [convert_to_one_hot_label(l,args.NUM_CLASSES) for l in labels]
    input_ids=[]
    attention_masks = []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=args.MAX_LEN, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert numpy arrays to pytorch tensors
    inputs = torch.cat(input_ids, dim=0)
    masks =  torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels,dtype=torch.float32)

    print("shape of labels = ", labels.shape)
    # Create the DataLoader for the given set (iterator to save memory)
    data = TensorDataset(inputs, masks, labels)
    if random_sampling: # train data
        sampler = RandomSampler(data)
    else: # dev and test data
        sampler = SequentialSampler(data)
    
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.BATCH_SIZE)
    return dataloader


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
            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
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
      
        ########## VALIDATION ##########
        t0 = time.time()
        model.eval() # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        eval_loss, eval_accuracy, nb_eval_examples = 0, 0, 0      # Tracking variables 

        # Tracking variables 
        total_eval_accuracy, total_eval_loss, nb_eval_steps = 0, 0, 0

        # Evaluate data for one epoch
        for batch in dev_dataloader:
            # Unpack the inputs from the dataloader object after adding batch to GPU
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():        
                # Forward pass, calculate logit predictions (output values prior to applying an activation function)
                (loss, logits) = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            logits = logits.detach().cpu().numpy() # Move logits to CPU
            label_ids = b_labels.to('cpu').numpy() # Move labels to CPU
            #print("label shape = ", label_ids.shape)
            total_eval_loss += loss.item() # Accumulate the validation loss.
            total_eval_accuracy += flat_accuracy(logits, label_ids) # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches.

            nb_eval_steps += 1 # Track the number of batches

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)
        if verbose: 
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Report the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(dev_dataloader)
        if verbose:
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        if verbose:
            print("  Validation took: {:}".format(validation_time))  
      
        # Record all statistics from this epoch.
        #avg_val_accuracy = 0
        #avg_val_loss = 0
        #validation_time = 0
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Train Loss': avg_train_loss,
                'Valid Loss': avg_val_loss,
                'Valid Acc': avg_val_accuracy,
                'Train Time': training_time,
                'Valid Time': validation_time
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

    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    return df_stats



def run(args):

    models = {
    "english_base_cased": ('bert-base-cased',False),
    "english_large_cased": ('bert-large-cased',False),
    "english_base_uncased": ('bert-base-uncased',True),
    "english_large_uncased": ('bert-large-uncased',True),
    "multilingual": ('bert-base-multilingual-cased',False),
    "finnish_cased": ('TurkuNLP/bert-base-finnish-cased-v1',False),
    "finnish_uncased": ('TurkuNLP/bert-base-finnish-uncased-v1',False)
    }

    args.save_dir = args.save_dir+args.language+"-"+args.BERT_MODEL

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"


    if args.use_pretrained:
        print("Using pre-trained bert : ",args.pretrained_path)
        bert_model = args.pretrained_path
        lowercase = False
    else:
        bert_model = models[args.BERT_MODEL][0]
        lowercase = models[args.BERT_MODEL][1]

    #print('Number of sentences train set: {:,}'.format(len(X_train)))
    #print('Number of sentences dev set:   {:,}'.format(len(X_dev)))
    #print('Number of sentences test set:  {:,}'.format(len(X_test)))
    #print("Number of classes: {}".format(args.NUM_CLASSES))

    config = BertConfig.from_pretrained(
        bert_model,
        num_labels=args.NUM_CLASSES,
    )

    model = BertForMultiLabelClassification.from_pretrained(
        bert_model,
        config=config
    )

    model.to(args.device)
    print_model_params(model)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=lowercase)


    if args.task=="train":

        data_path = args.data_path 
        df_dataset = pd.read_csv(data_path, delimiter='\t', header=None, names=['sentence','label']).sample(frac=1, random_state=args.SEED)
        df_train, df_dev, df_test = np.split(df_dataset, [int(args.PCTG_TRAIN*len(df_dataset)), int((1-args.PCTG_TEST)*len(df_dataset))])

        X_train = df_train.sentence.values
        y_train = df_train.label.values
        X_dev = df_dev.sentence.values
        y_dev = df_dev.label.values
        X_test = df_test.sentence.values
        y_test = df_test.label.values
        
        train_dataloader      = prepare_data(X_train, y_train, tokenizer, args, True)
        dev_dataloader        = prepare_data(X_dev, y_dev, tokenizer, args, False)
        prediction_dataloader = prepare_data(X_test, y_test, tokenizer, args, False)

        adam         = AdamW(model.parameters(), lr=args.LEARN_RATE, eps=args.EPSILON)
        total_steps  = len(train_dataloader) * args.EPOCHS # Total number of training steps is nb_batches times nb_epochs.
        linear_sch   = get_linear_schedule_with_warmup(adam, num_warmup_steps=args.nb_warmup_steps, num_training_steps=total_steps)

        training_stats = train(model=model, tokenizer = tokenizer, optimizer=adam, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, 
                            scheduler=linear_sch, args= args)
        print(training_stats)
        print("\n\nPredicting on Test split\n")
        predictions, true_labels = predict(model, prediction_dataloader, args)
        mf1, acc = evaluate(predictions, true_labels, verbose=True)
        print("f1:", mf1, "  acc:",acc)


    elif args.task == "test":

        df_test = pd.read_csv(args.test_path, delimiter='\t', header=None, names=['sentence','label']).sample(frac=1, random_state=args.SEED)
        X_test = df_test.sentence.values
        y_test = df_test.label.values
        prediction_dataloader = prepare_data(X_test, y_test, tokenizer, args, False)

        predictions, true_labels = predict(model, prediction_dataloader, args)
        mf1, acc = evaluate(predictions, true_labels, verbose=True)
        print("\n\nPredicting on Test split\n")
        print("f1:", mf1, "  acc:",acc)

    elif args.task == "gen_pseudo":

        df_dataset = pd.read_csv(args.data_path, delimiter='\t', header=None, names=['sentence','label']).sample(frac=1, random_state=args.SEED)

        X_all = df_dataset.sentence.values
        Y_all = df_dataset.label.values
        
        all_dataloader = prepare_data(X_all, Y_all, tokenizer, args, False)
        predictions, true_labels = predict(model, all_dataloader, args)
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        generate_pseudo_data(flat_predictions,Y_all,args.annotated_file,X_all)

        mf1, acc = evaluate(predictions, true_labels, verbose=True)
        print("f1:", mf1, "  acc:",acc)


    if args.CROSS_VALIDATION:
        cross_validation(args, tokenizer)

def set_seeds(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="train")

    parser.add_argument("--save_dir", type=str, default="/nobackup3/wb/xed/")
    parser.add_argument("--data_path", type=str, default="AnnotatedData/pseudo-multilingual-ekman-fi-annotated.tsv")
    parser.add_argument("--test_path", type=str, default="AnnotatedData/ekman-fi-annotated.tsv")


    parser.add_argument("--SEED", type=int, default=12345)
    parser.add_argument("--EPOCHS", type=int, default=3)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--language", type=str, default="english")

    # hyper parameters
    parser.add_argument("--BERT_MODEL", type=str, help="multilingual , english_base_cased , english_large_cased , english_base_uncased, english_large_uncased, finnish_cased, finnish_uncased",
                     default="multilingual")

    parser.add_argument("--use_pretrained",  type=bool, default=False)
    parser.add_argument("--pretrained_path",  type=str, default="/nobackup3/wb/xed/english-multilingual/checkpoint-3")
    parser.add_argument("--annotated_file", type=str, default="AnnotatedData/pseudo-multilingual-ekman-fi-annotated.tsv")

    parser.add_argument("--THRESHOLD",  type=int, default=0.3)
    parser.add_argument("--BATCH_SIZE",  type=int, default=96)
    parser.add_argument("--MAX_LEN",  type=int, default=48)
    parser.add_argument("--nb_warmup_steps",  type=int, default=0)
    parser.add_argument("--LEARN_RATE", type=float, default=2e-5)
    parser.add_argument("--EPSILON",  type=float, default=1e-8)

    parser.add_argument("--CROSS_VALIDATION",  type=bool, default=False)
    parser.add_argument("--CHANGE_SPLIT",  type=bool, default=False)
    parser.add_argument("--NUM_FOLDS",  type=int, default=5)

    parser.add_argument("--PCTG_TRAIN",  type=float, default=0.75)
    parser.add_argument("--PCTG_DEV",  type=float, default=0.15)
    parser.add_argument("--PCTG_TEST",  type=float, default=0.10)

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