# Set the save directory to where you want to save the models
SAVE_PATH='outputs'

echo "Train BERT Model"
python bert.py \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=bert_

sleep 5
echo "Test BERT Model"
# Set the save directory to where you want to save the models
python bert.py \
    --task=test \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=bert_ \
    --use_pretrained=True \
    --checkpoint=10






