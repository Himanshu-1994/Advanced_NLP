import numpy as np
import os
import pandas as pd

PCTG_TRAIN = 0.75
PCTG_TEST = 0.15
PCTG_DEV = 0.10

data_path = "ekman-en-annotated.tsv"
df_dataset = pd.read_csv(data_path, delimiter='\t', header=None, names=['sentence','label']).sample(frac=1, random_state=42)
df_train, df_dev, df_test = np.split(df_dataset, [int(PCTG_TRAIN*len(df_dataset)), int((1-PCTG_TEST)*len(df_dataset))])

folder = "ekman"
if not os.path.exists(folder):
    os.makedirs(folder)

data = "ekman-en-annotated"
df_train.to_csv(os.path.join(folder,data+"_train.tsv"), sep='\t', encoding='utf-8', index=False)
df_dev.to_csv(os.path.join(folder,data+"_dev.tsv"), sep='\t', encoding='utf-8', index=False)
df_test.to_csv(os.path.join(folder,data+"_test.tsv"), sep='\t', encoding='utf-8',index=False)
