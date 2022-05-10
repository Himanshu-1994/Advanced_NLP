# Emotion Detection
This repositry contains the code for Emotion Detection Project
as part of Advanced Natural Language Processing Course (CS769) UWMadison.

## Run instructions:

1. Create a virtual environment and install *torch* and the libraries given in *min_requirements.txt*
    
    ```
    $ python3 -m venv /path/to/new/virtual/environment
    $ source /path/to/new/virtual/environment/bin/activate
    $ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 
    $ pip3 install -r min_requirements.txt
    ```

2. If you want to modify the local tmpdir for storing downloaded transformer models from Hugging Face,
   use one of the following: 
    ```
    import os
    os.environ['TRANSFORMERS_CACHE'] = PATH_TO_CACHE_DIR
    OR
    $ export TRANSFORMERS_CACHE=PATH_TO_CACHE_DIR
    ```

3. Modify the *SAVE_PATH* in each .sh script to set the default directory for saving trained models.

    *The following are the scripts for running various experiments presented in this Project.*

- To train and test the **best performing model** **GPT2** on goEmotions dataset with best hyperparameter settings.

    `$ sh run_gpt2_demoji.sh`

- To train and test the reimplementation of baseline **BERT** model on goEmotions dataset.

    `$ sh run_bert.sh`

- To train and test the **GPT2** model without any augmentation

    `$ sh run_gpt2.sh`

- To train and test the **T5** model.

    `$ sh run_t5.sh`

- To run any other experiment, modify the scripts accordingly.
  
  For example to train the **GPT-2** model with **PEGASUS** data augmentation
  
    ```
    export SAVE_PATH='/nobackup3/himanshu/outputs'
    python gpt2.py \
        --SEED=100 \
        --save_dir=${SAVE_PATH} \
        --suff=gpt2_ \
        --train_path=data/original/augmented/train_paraphrase_samplelow.tsv \
        --dev_path=data/original/dev.tsv \
        --test_path=data/original/test.tsv
    ```

- To train and test the model on ekman split of GoEmotions.

    `$ sh run_gpt2_demoji_ekman.sh`

- To train and test the model on group split of GoEmotions.

    `$ sh run_gpt2_demoji_grouped.sh`

## Data Augmentation

- Creating emoji data
    ```
    $ python emoji_list.py
    $ demoji data/original/train.tsv > data/original/train_demoji.tsv
    $ demoji data/original/dev.tsv > data/original/dev_demoji.tsv
    $ demoji data/original/test.tsv > data/original/test_demoji.tsv
    ```

- PEGASUS paraphrashing data generation
    ```
    $ python pegasus_paraphrase.py --input=data/original/train.tsv --output=data/original/augmented/train_paraphrase_samplelow.tsv
    ```

- Backtranslation and contextual data generation
    ```
    $ python backtranslate_contextual.py --input=data/original/train.tsv --output=data/original/augmented/train_demoji_backtranslate_contextual.tsv
    ```

- Synonym Replacement data generation
    ```
    $ python syn_replace.py --input=data/original/train.tsv --output=data/original/augmented/train_augnew.tsv
    ```
- EDA data generation:
  Inspired from https://github.com/jasonwei20/eda_nlp
    ```
    $ python augment.py --input=data/original/train.tsv --output=data/original/augmented/train_eda.tsv
    ```

## Cross-lingual transfer

- To run the cross-lingual experiment.(Ignore intermediate outputs)
    **Final result** is stored in $SAVE_DIR/goemotions-ekman/bert-finnish-final_seed_100/results_test

    `$ sh run_crosslingual.sh`