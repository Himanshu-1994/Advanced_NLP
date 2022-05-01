import os
import csv
import pdb

class DataDivider:
    def __init__(self, file_path, target_path):
        self.file_path = file_path
        self.target_path = target_path

    def divide_tsv(self):
        with open(self.file_path) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for line in tsv_file:
                labels = line[1].split(',')
                for label in labels:
                    with open(self.target_path + '/' + label + '.tsv', 'a') as out_file:
                        tsv_writer = csv.writer(out_file, delimiter='\t')
                        tsv_writer.writerow(line)

if __name__ == "__main__":
    train_path = '/Users/kaushikkota/ms_cs_uw_madison/cs_769_nlp/Advanced_NLP/data/original/train.tsv'
    target_path = '/Users/kaushikkota/ms_cs_uw_madison/cs_769_nlp/Advanced_NLP/utils/ClassBasedData'
    data_loader = DataDivider(train_path, target_path)
    data_loader.divide_tsv()




