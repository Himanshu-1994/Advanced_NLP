import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import argparse

MODEL_NAME = 'tuner007/pegasus_paraphrase'
torch_device = "cuda"
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(torch_device)

def get_response(input_text, num_return_sequences):
    batch = tokenizer.prepare_seq2seq_batch([input_text],
                                            truncation=True,
                                            padding='longest',
                                            return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,
                                num_beams=num_return_sequences,
                                num_return_sequences=num_return_sequences,
                                temperature=1.5).to(torch_device)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=True, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=True, type=int, help="number of augmented sentences per original sentence")
args = ap.parse_args()


input_file = args.input
output_file = args.output
num_return_sequences = args.num_aug

writer = open(output_file, 'w')
lines = open(input_file, 'r').readlines()
classes_to_sample = [3,5,6,12,16,19,21,23]


for i, line in enumerate(lines):
    parts = line[:-1].split('\t')
    label = parts[0]
    labs = list(map(int, label.split(",")))
    sentence = parts[1]

    #Write 1 sentence to every line
    writer.write(sentence + "\t" + label + '\n')

    if not any(x in labs for x in classes_to_sample):
        continue

    aug_sentences = get_response(sentence,num_return_sequences)

    for aug_sentence in aug_sentences:
        if len(aug_sentence)==0:
            continue

        new_lab = ""
        for l in labs:
            if l in classes_to_sample:
                new_lab+=str(l)+","
        label = new_lab[:-1]
        writer.write(aug_sentence + "\t" + label + '\n')
writer.close()
print("generated augmented sentences with eda for " + input_file + " to " + output_file)
