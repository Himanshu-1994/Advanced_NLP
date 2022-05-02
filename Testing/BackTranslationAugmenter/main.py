import csv
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action
import nlpaug.augmenter.word as naw


if __name__ == "__main__":
    labels_count = 28
    dir_path = '/Users/kaushikkota/ms_cs_uw_madison/cs_769_nlp/Advanced_NLP/utils/ClassBasedData'
    target_path = '/Users/kaushikkota/ms_cs_uw_madison/cs_769_nlp/Advanced_NLP/Testing/BackTranslationAugmenter/data'
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de',
        to_model_name='facebook/wmt19-de-en'
    )

    '''
    from_model_dir = os.path.join('/Users/kaushikkota/ms_cs_uw_madison/cs_769_nlp/Advanced_NLP/Testing/BackTranslationAugmenter/wmt19.en-de.joined-dict.ensemble')
    to_model_dir = os.path.join('/Users/kaushikkota/ms_cs_uw_madison/cs_769_nlp/Advanced_NLP/Testing/BackTranslationAugmenter/wmt19.de-en.joined-dict.ensemble')

    back_translation_aug = naw.BackTranslationAug(
        from_model_name=from_model_dir,
        to_model_name=to_model_dir)
    '''
    for i in range(labels_count):
        with open(dir_path + '/' + str(i) + '.tsv') as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for line in tsv_file:
                sentence = line[0]
                aug_sentence = back_translation_aug.augment(sentence)
                line[0] = aug_sentence
                with open(target_path + '/train.tsv', 'a') as out_file:
                    tsv_writer = csv.writer(out_file, delimiter='\t')
                    tsv_writer.writerow(line)



