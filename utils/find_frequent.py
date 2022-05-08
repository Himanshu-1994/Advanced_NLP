import numpy as np
import os
import pandas as pd
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import pickle
from itertools import chain
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
import string
import fightin-words as fw
import sklearn.feature_extraction.text as sk_text

cv = sk_text.CountVectorizer(max_features=15000)
prior = 0.05

def clean_text(text):
    """
    Function to clean the text.
    
    Parameters:
    text: the raw text as a string value that needs to be cleaned
    
    Returns:
    cleaned_text: the cleaned text as string
    """
    # convert to lower case
    cleaned_text = text.casefold()
    # remove HTML tags
    html_pattern = re.compile('<.*?>')
    cleaned_text = re.sub(html_pattern, '', cleaned_text)
    # remove punctuations
    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))
    
    return cleaned_text.strip()



nltk.download("stopwords")
nltk.download('punkt')
stopwords_ls = list(set(stopwords.words("english")))
stopwords_ls = [clean_text(word) for word in stopwords_ls]

def findtop(data):
    
    data = [clean_text(s) for s in data]
    tokens = [word_tokenize(s) for s in data]
    flat_tokens = list(chain.from_iterable(tokens))

    flat_tokens = [s for s in flat_tokens if not s in stopwords_ls]
    freq = FreqDist(flat_tokens)
    top = freq.most_common(20)
    return top

input_file = "XED/AnnotatedData/ekman-en-annotated.tsv"

labelwise = defaultdict(list)

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

    for line in lines:

        line = line.strip()
        vals = line.split("\t")
        text = vals[0]
        label = list(map(int, vals[1].split(",")))

        for l in label:
            labelwise[l].append(text)

topterms = defaultdict(list)

for key in labelwise:

    topterms[key] = findtop(labelwise[key])

with open("frequent-terms",'wb') as f:
    pickle.dump(topterms,f)