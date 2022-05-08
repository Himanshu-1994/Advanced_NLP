# Emotion Detection
This repositry contains the code for Emotion Detection Project
as part of Advanced Natural Language Processing Course (CS769) UWMadison.

## Run instructions:

0. Create a virtual environment and install the libraries given in *min_requirements.txt*
    
    ```
    $ python3 -m venv /path/to/new/virtual/environment
    $ source /path/to/new/virtual/environment/bin/activate
    $ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 
    $ pip3 install -r min_requirements.txt
    ```

1. If you want to modify the local tmpdir for storing downloaded models,
uncomment the first line : `os.environ['TRANSFORMERS_CACHE'] = PATH_TO_CACHE_DIR`

2. Modify the *SAVE_PATH* in each .sh script to set the default directory for saving trained models.

*The following are the scripts for running various experiments presented in the Report.*

- To train and test the **best performing model** **GPT2** on goEmotions dataset with best hyperparameter settings.

    `$ sh run_gpt2_demoji.sh`

- To train and test the reimplementation of baseline **BERT** model on goEmotions dataset.

    `$ sh run_bert.sh`

- To train and test the **GPT2** model without any augmentation

    `$ sh run_gpt2.sh`

- To train and test the **T5** model.

    `$ sh run_t5.sh`

- To run any other experiment, modify the scripts accordingly.
  
  ### For example to train the GPT-2 model with PEGASUS data augmentation
  
  ```
  export SAVE_PATH='/nobackup3/himanshu/xed/outputs'
  python gpt2.py \
    --SEED=100 \
    --save_dir=${SAVE_PATH} \
    --suff=gpt2_ \
    --train_path=data/original/augmented/train_paraphrase_samplelow.tsv \
    --dev_path=data/original/dev.tsv \
    --test_path=data/original/test.tsv
  ```


## Data Augmentation

- Creating emoji list
    `$ python emoji_list.py`