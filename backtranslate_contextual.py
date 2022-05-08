from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re,string
from nltk.corpus import wordnet
import argparse
import nltk
import random
import os

if not os.path.exists("data/nltk-data"):
    nltk.data.path.append('./nltk-data/')
    nltk.download('wordnet')

import demoji
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action
import nlpaug.augmenter.word as naw



ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--aug_type", required=False, type=str, help="Aug type", default="backtranslate")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
args = ap.parse_args()

stop_words = stopwords.words('english')
special_words = ['[','NAME',']','RELIGION']


def data_augmentation(input_file,output_file,args):
    writer = open(output_file, 'w')
    lines = open(input_file, 'r').readlines()

    classes_to_sample = [3,5,6,12,16,19,21,23]
    #if args.aug_type=="contextual":
    context_aug = naw.ContextualWordEmbsAug(model_path='bert-base-cased', action="substitute",device="cuda")

    #if args.aug_type=="backtranslate":
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de',
        to_model_name='facebook/wmt19-de-en',
        device="cuda",
    )

    to_augment = []
    to_labels = []
    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        label = parts[1]
        labs = list(map(int, label.split(",")))
        sentence = parts[0]

        #Write 1 sentence to every line
        emojis = demoji.findall(sentence)
        sentence = demoji.replace_with_desc(sentence,":")
        writer.write(sentence + "\t" + label + '\n')
        if len(emojis)>0:
            continue

        if not any(x in labs for x in classes_to_sample):
            continue

        to_augment.append(sentence)
        to_labels.append(labs)

    print("Starting back translation on total=",len(to_augment))
    aug_sentences = back_translation_aug.augment(to_augment)
    print("Done translation")
    
    print("Starting contextual augmentation on total=",len(to_augment))
    aug_sentences_2 = context_aug.augment(to_augment)
    print("Done translation")
    
    for k,aug_sentence in enumerate(aug_sentences):
        if aug_sentence=="":
            continue

        new_lab = ""
        labs = to_labels[k]

        for l in labs:
            if l in classes_to_sample:
                new_lab+=str(l)+","
        label = new_lab[:-1]
        writer.write(aug_sentence + "\t" + label + '\n')
        writer.write(aug_sentences_2[k] + "\t" + label + '\n')

        if k%100==0:
            print("Done lines: ",i," in: ",len(lines))
            writer.flush()

    writer.close()
    print("generated augmented sentences with backtranslation and contextual augmentation for " + input_file + " to " + output_file)


print(args)
output_file = args.output
input_file = args.input
data_augmentation(input_file,output_file,args)