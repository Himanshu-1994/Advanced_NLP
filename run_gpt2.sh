# Set the save directory to where you want to save the models
SAVE_PATH='/nobackup3/himanshu/xed/outputs'

echo "Train GPT2 Model"
python gpt2.py \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=gpt2_ \
    --EPOCHS=3

sleep 5
echo "Test GPT2 Model"
# Set the save directory to where you want to save the models
python gpt2.py \
    --task=test \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=gpt2_ \
    --use_pretrained=True \
    --checkpoint=3