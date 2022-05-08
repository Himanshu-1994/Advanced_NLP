import os
import copy
import numpy as np
examples = []

input_file = "XED/AnnotatedData/ekman/pseudo_labels_finnish_test.tsv"
output_file = "XED/AnnotatedData/ekman/pseudo_labels_finnish_test_clean.tsv"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

    for (i, line) in enumerate(lines):
        line = line.strip()
        vals = line.split("\t")
        text = vals[0]
        if len(vals)==1 or vals[1]=="":
            continue

        examples.append(text+'\t'+vals[1]+'\n')

with open(output_file,"w", encoding="utf-8") as f:
    for line in examples:
        f.write(line)
    
