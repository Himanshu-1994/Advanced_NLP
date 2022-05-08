# Set the save directory to where you want to save the models
SAVE_PATH='/nobackup3/himanshu/xed/outputs'

echo "Train t5 Model"
python t5.py \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=t5_ \
    --EPOCHS=1

sleep 5
echo "Test t5 Model"
# Set the save directory to where you want to save the models
python t5.py \
    --task=test \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=t5_ \
    --use_pretrained=True \
    --checkpoint=1