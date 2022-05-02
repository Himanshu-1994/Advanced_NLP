import csv
import os
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action
import nlpaug.augmenter.word as naw


if __name__ == "__main__":
    os.environ["MODEL_DIR"] = '../model'
    labels_count = 28
    dir_path = '/Users/kaushikkota/ms_cs_uw_madison/cs_769_nlp/Advanced_NLP/utils/ClassBasedData'
    target_path = '/Users/kaushikkota/ms_cs_uw_madison/cs_769_nlp/Advanced_NLP/Testing/ContextualWordEmbeddingAugmenter/data'

    # Can use other models BERT, DistilBERT, RoBERTA or XLNet
    context_aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

    for i in range(labels_count):
        with open(dir_path + '/' + str(i) + '.tsv') as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for line in tsv_file:
                sentence = line[0]
                aug_sentence = context_aug.augment(sentence)
                line[0] = aug_sentence
                with open(target_path + '/train.tsv', 'a') as out_file:
                    tsv_writer = csv.writer(out_file, delimiter='\t')
                    tsv_writer.writerow(line)




