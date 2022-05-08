import pandas as pd
import os

train_path = "data/"+"original"+"/train.tsv"
dev_path = "data/"+"original"+"/dev.tsv"
test_path = "data/"+"original"+"/test.tsv"

df_train = pd.read_csv(train_path, delimiter='\t', header=None, names=['sentence','label','extra']).sample(frac=1, random_state=0)
df_dev = pd.read_csv(dev_path, delimiter='\t', header=None, names=['sentence','label','extra']).sample(frac=1, random_state=0)
df_test = pd.read_csv(test_path, delimiter='\t', header=None, names=['sentence','label','extra']).sample(frac=1, random_state=0)

df_train = df_train[['sentence']]
df_dev = df_dev[['sentence']]
df_test = df_test[['sentence']]
df_data = pd.concat([df_train, df_dev, df_test]).sample(frac=1, random_state=0)

df_data.to_csv(os.path.join("data/original/lmtrain.tsv"), sep='\t', encoding='utf-8', index=False)

