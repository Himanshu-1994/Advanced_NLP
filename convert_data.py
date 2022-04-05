import os
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset

# {neutral:4,anger:0,disgust:1,fear:2,joy:3,sadness:5,surprise:6}
map_label = {0:4,1:0,3:1,4:2,5:3,6:5,7:6}
# Exempt {anticipation,trust}
exempt = [2,8]

examples = []

input_file = "XED/AnnotatedData/en-annotated.tsv"
output_file = "XED/AnnotatedData/ekman-en-annotated.tsv"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

    for (i, line) in enumerate(lines):
        line = line.strip()
        vals = line.split("\t")
        text = vals[0]
        label = list(map(int, vals[1].split(",")))
        label = [map_label[i] for i in label if i not in exempt]

        if len(label)==0:
            continue

        labstrn = ",".join(str(i) for i in label)
        examples.append(text+'\t'+labstrn+'\n')

with open(output_file,"w", encoding="utf-8") as f:
    for line in examples:
        f.write(line)
    
