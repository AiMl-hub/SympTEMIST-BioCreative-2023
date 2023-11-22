# TODO change results path

# cd medprocner_evaluation_library-main
# python3 medprocner_evaluation.py -r ../data/subtask1-ner/gold_standard.tsv -p ../results/model-bsc-bio-ehr-es/test.tsv -t ner -o scores/

import glob
import os
import pandas as pd

from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch


DATA_PATH = "./data/subtask1-ner/"
TXT_FILES_PATH = f"{DATA_PATH}test_txt/"
RESULTS_PATH = "./results"

# Load your label mapping (id2label and label2id) here
id2label = {0: "O", 1: "B-SINTOMA", 2: "I-SINTOMA"}  # Replace with your label mapping
label2id = {v: k for k, v in id2label.items()}  # Convert id2label to label2id

# Load the model and tokenizer
# TODO change model path
model_checkpoint = "./full_output/model-intfloat/multilingual-e5-large/dataset-data/subtask1-ner/processed_SympTEMIST/seed-42/epochs-70"

inference_model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


# Read test dataset
test_df = pd.read_csv("data/subtask1-ner/test_SympTEMIST_dataset.csv")

results_dict = {}
data_dict = {}
# Iterate over files
for _, row in test_df.iterrows():
    results_ann_id = 0
    ann_id = 0
    filename = row["filename"]
    file_path = f"{TXT_FILES_PATH}{filename}.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        # Read the entire contents of the file
        text = file.read()

    tokenized_inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True
    )

    encodings = tokenizer(
        text, return_offsets_mapping=True, padding="max_length", truncation=True
    )

    # Get the model prediction
    with torch.no_grad():
        logits = inference_model(**tokenized_inputs).logits
        print(logits.shape)

    predictions = torch.argmax(logits, dim=2)
    labels = predictions[0].numpy()

    prev_end_pos = []
    for i, (token_id, pos) in enumerate(
        zip(encodings["input_ids"], encodings["offset_mapping"])
    ):
        prev_end_pos.append(pos[1])

        if labels[i] == 1:
            if labels[i - 1] != 1:
                ann_id += 1
                key = filename + str(ann_id)
                data_dict[key] = [
                    filename,
                    "T" + str(ann_id),
                    "SINTOMA",
                    pos[0],
                    pos[1],
                    text[pos[0] : pos[1]],
                ]
            else:
                if pos[0] != prev_end_pos[i - 1]:
                    data_dict[key][4] = pos[1]
                    # To add space only if they are separate words (tokens)
                    data_dict[key][5] += " " + text[pos[0] : pos[1]]
                else:
                    data_dict[key][4] = pos[1]
                    # The current and the previous tokens are one word (don't add space)
                    data_dict[key][5] += text[pos[0] : pos[1]]
        elif labels[i] == 2:
            if pos[0] != prev_end_pos[i - 1]:
                data_dict[key][4] = pos[1]
                # To add space only if they are separate words (tokens)
                data_dict[key][5] += " " + text[pos[0] : pos[1]]

            else:
                data_dict[key][4] = pos[1]
                # The current and the previous tokens are one word (don't add space)
                data_dict[key][5] += text[pos[0] : pos[1]]

        # if labels[i] != 0:
        #     if labels[i] == 1:
        #         ann_id += 1
        #     elif (
        #         labels[i] == 2 and labels[i - 1] == 0
        #     ):  # that means a token is B- but it is incorrectly labeled as I-
        #         ann_id += 1
        #     key = filename + "_" + str(ann_id)
        #     if key not in data_dict.keys():
        #         data_dict[key] = [
        #             filename,
        #             "T" + str(ann_id),
        #             "SINTOMA",
        #             pos[0],
        #             pos[1],
        #             text[pos[0] : pos[1]],
        #         ]
        #     else:
        #         data_dict[key][4] = pos[1]
        #         if pos[0] != prev_end_pos[i - 1]:
        #             # To add space only if they are separate words (tokens)
        #             data_dict[key][5] += " " + text[pos[0] : pos[1]]

        #         elif pos[0] == prev_end_pos[i - 1]:
        #             # The current and the previous tokens are one word (don't add space)
        #             data_dict[key][5] += text[pos[0] : pos[1]]

        results_ann_id += 1
        results_key = filename + str(results_ann_id)
        results_dict[results_key] = [
            filename,
            labels[i],
            pos[0],
            pos[1],
            text[pos[0] : pos[1]],
        ]


def create_csv_files(csv_file_path, dict):
    """Save data in TSV format

    Args:
        csv_file_path (string): file path to save the CSV file
        dict (dictionary): dictionary with the data to be saved
    """
    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame(dict)
    df = df.T
    df.columns = ["filename", "ann_id", "label", "start_span", "end_span", "text"]
    # Save the DataFrame to a TSV file
    df.to_csv(csv_file_path, index=False, sep="\t")


model_name = model_checkpoint.split("/")[2]
if not os.path.exists(f"{RESULTS_PATH}/{model_name}"):
    os.makedirs(f"{RESULTS_PATH}/{model_name}")

create_csv_files(f"{RESULTS_PATH}/{model_name}/test.tsv", data_dict)

df = pd.DataFrame(results_dict)
df = df.T
df.columns = ["filename", "label", "start_span", "end_span", "text"]
df.to_csv(f"{RESULTS_PATH}/{model_name}/results_test.tsv", index=False, sep="\t")

print("Inference Completed!")
