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
args = ap.parse_args()

stop_words = stopwords.words('english')
special_words = ['[','NAME',']','RELIGION']

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if word != l.name():
                synonyms.append(l.name())
    return synonyms

def synonym_replacement(text):
    #sentences = [text]
    sentences = []
    text = re.sub("\[NAME\]"," [NAME] ",text)
    text = re.sub("\[RELIGION\]"," [RELIGION] ",text)
    words = word_tokenize(text)
    filter_words = [word for word in words if (not word.lower() in stop_words) \
        and (not word in special_words)]
    
    tokens = list(filter(lambda token: token not in string.punctuation, filter_words))

    word_bag = list(set(tokens))
    n = 1  #  each generated text will have two words replaced with their respective synonym 
    combined_bag = [word_bag[i * n:(i + 1) * n] for i in range((len(word_bag) + n - 1) // n)]
    to_exchange = combined_bag
    payload = text
    try:
        for words in to_exchange:
            payload = text
            for word in words:
                similar_words = get_synonyms(word)
                if similar_words is not None and len(similar_words)>0:
                    similar_words = [re.sub('[^A-Za-z ]+', ' ', sent) for sent in similar_words]
                    synonym = word
                    tries = 0
                    while synonym==word:
                        synonym = random.choice(list(similar_words))
                        payload = re.sub(word, synonym, payload)
                        tries+=1
                        if tries>10:
                            break

            sentences.append(payload.strip())
            
        sentences = list(set(sentences))
    except Exception as e:
        print(e)
    return sentences


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def data_augmentation(input_file,output_file,args):
    writer = open(output_file, 'w')
    lines = open(input_file, 'r').readlines()

    classes_to_sample = [3,5,6,12,16,19,21,23]

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

        aug_sentences = synonym_replacement(sentence)

        if len(aug_sentences)==0:
            continue
        for aug_sentence in aug_sentences:
            if aug_sentence=="":
                continue

            new_lab = ""
            for l in labs:
                if l in classes_to_sample:
                    new_lab+=str(l)+","
            label = new_lab[:-1]
            writer.write(aug_sentence + "\t" + label + '\n')
        writer.flush()

        if i%100==0:
            print("Done lines: ",i," in: ",len(lines))
    writer.close()
    print("generated augmented sentences with eda for " + input_file + " to " + output_file)


print(args)
output_file = args.output
input_file = args.input
data_augmentation(input_file,output_file,args)