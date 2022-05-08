# Set the save directory to where you want to save the models
SAVE_PATH='outputs'

echo "Train GPT2 Model with Emoji removal"
python gpt2.py \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=gpt2_demoji_ \
    --demoji=True \
    --train_path=data/original/train_demoji.tsv \
    --dev_path=data/original/dev_demoji.tsv \
    --test_path=data/original/test_demoji.tsv

sleep 5
echo "Test GPT2 Model with Emoji removal"
# Set the save directory to where you want to save the models
python gpt2.py \
    --task=test \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=gpt2_demoji_ \
    --demoji=True \
    --use_pretrained=True \
    --checkpoint=3 \
    --train_path=data/original/train_demoji.tsv \
    --dev_path=data/original/dev_demoji.tsv \
    --test_path=data/original/test_demoji.tsv