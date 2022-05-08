import os,sys
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset
from collections import defaultdict
# {neutral:4,anger:0,disgust:1,fear:2,joy:3,sadness:5,surprise:6}
map_label = {0:4,1:0,3:1,4:2,5:3,6:5,7:6}
# Exempt {anticipation,trust}
exempt = [2,8]

examples = []

#input_file = "data/original/augmented/train_demoji_backtranslate_1.tsv"
#input_file = "data/original/augmented/train_paraphrase_samplelow.tsv"
input_file = "data/original/train.tsv"
#output_file = "data/original/train_stat.tsv"

total_21 = 0
total_0 = 0
joint = 0
counts = defaultdict(int)
total=0.0
s=0

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

    for (i, line) in enumerate(lines):
        line = line.strip()
        vals = line.split("\t")
        text = vals[0]
        label = list(map(int, vals[1].split(",")))
        for h in label:
            counts[h]+=1

        if 27 in label and len(label)>1:
            s+=1
        if 0 in label and 27 in label:
            joint+=1
        if 0 in label:
            total_0+=1
        if 21 in label:
            total_21+=1

        #if len(label)>1:
        #    count+=1
        #total+=1

#print(s)
#print("\ntotal_0 %d,total_21 %d, joint %d\n"%(total_0,total_21,joint))
res = 0
ratios = []
#total = sum(list(counts))
for key in counts:
    total+=counts[key]
print("Total = %d"%(total))
for i in range(28):
    #print("label%d:%f"%(i,counts[i]/total))
    print(i,counts[i])
    ratios.append(total/counts[i])
    #res += counts[i]/total
sys.exit()
with open("perclassnew.txt",'w') as f:
    for i in range(28):
         #in counts:
        f.write(str(i)+": "+str(counts[i])+"\n")

num = sum(ratios)
ratios = [r/num for r in ratios]
ratios = [1 for _ in range(28)]
ratios[21] = 3
ratios[0] = 0.6
ratios[23] = 2.5
#np.save('ratios',ratios)
print(ratios)
