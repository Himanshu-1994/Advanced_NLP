import os
import copy
import numpy as np

examples = []

input_file = "data/original/test.tsv"
output_file = "data/original/augmented/test.tsv"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

    for (i, line) in enumerate(lines):
        line = line.strip()
        vals = line.split("\t")
        text = vals[0]
        labels = vals[1]
        examples.append(labels+'\t'+text+'\n')

with open(output_file,"w", encoding="utf-8") as f:
    for line in examples:
        f.write(line)
    
