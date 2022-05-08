import numpy as np 
import demoji
import pickle

emoji_dict = {}
input_file = "data/original/train.tsv"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

    for sentence in lines:
        s = demoji.findall(sentence)
        for em in s:
            if not em in emoji_dict:
                emoji_dict[em]=s[em]

emoji_list = []
for emoji in emoji_dict:
    emoji_list.append(":"+emoji_dict[emoji]+":")

with open("emoji_tokens.pkl",'wb') as f:
    pickle.dump(emoji_list,f)


