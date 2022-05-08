# Set the save directory to where you want to save the models
SAVE_PATH='outputs'

echo "Finetune multilingual-BERT Model on English data"
python bert.py \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=multibert_ \
    --arch=multilingual \
    --train_path=data/AnnotatedData/ekman-en-annotated_train.tsv \
    --type=ekman \
    --freeze=True \
    --EPOCHS=5

sleep 5
echo "Generate pseudo-labels in Finnish using multilingual-BERT"
# Set the save directory to where you want to save the models
python bert.py \
    --SEED=100 \
    --task=test \
    --save_dir=${SAVE_PATH} \
    --suff=multibert_ \
    --arch=multilingual \
    --use_pretrained=True \
    --test_path=data/AnnotatedData/ekman-fi-annotated_test.tsv \
    --type=ekman \
    --checkpoint=5 \
    --gen_labels=True

sleep 5
echo "Self-supervised Finnish-BERT model Language modelling objective"
python train_bert_lm.py \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=bert-finnish-lm_ \
    --arch=finnish_cased \
    --train_path=data/AnnotatedData/ekman-fi-annotated.tsv \
    --type=ekman \
    --EPOCHS=5

sleep 5
echo "Fine-tune Finnish-BERT model using pseudolabels"
python bert.py \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=bert-finnish-final_ \
    --use_pretrained=True \
    --pretrained_path=${SAVE_PATH}/goemotions-ekman/bert-finnish-lm_seed_100/checkpoint-5/ \
    --train_path=data/AnnotatedData/pseudo_labels_finnish_test_clean.tsv \
    --type=ekman \
    --freeze=True \
    --EPOCHS=3

sleep 5
echo "Test final Finnish-BERT model"
python bert.py \
    --SEED=100 \
    --task=test \
    --save_dir=${SAVE_PATH} \
    --suff=bert-finnish-final_ \
    --use_pretrained=True \
    --test_path=data/AnnotatedData/ekman-fi-annotated_test.tsv \
    --type=ekman \
    --checkpoint=3
