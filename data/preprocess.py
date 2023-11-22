"""This file preprocesses the data for the NER task."""


# 1 Make Changes in preprocess.py
# 2 I fix one model for testing, epochs = 2, I generate results with this one model.
# 3 I compare the results with eval_dataset.tsv
# I repeat step 1 -3 until i get desired results, by results here we mean the format of data.

# SOME CHANGES
# 1 brat to ConLL
# 2 stratified sampling
# 3 train (around 30% eval_f1)
import pandas as pd
import spacy
from datasets import (
    Dataset,
    DatasetDict,
    ClassLabel,
    Sequence,
    Value,
    Features,
)

DATA_PATH = "data/subtask1-ner/"
TXT_FILES_PATH = f"{DATA_PATH}txt/"

df = pd.read_csv(
    f"{DATA_PATH}tsv/symptemist_tsv_train_subtask1.tsv",
    sep="\t",
)

# Initialize variables for NER conversion
file_id = 0  # Initialize ID
filenames = []
id_sequence = []
token_sequence = []
ner_tags_sequence = []
start_end_sequence = []

# Initialize variables for faulty files
all_files_missed_filenames = []
all_files_missed_texts = []
all_files_missed_positive_tokens = []
all_files_missed_tokens = []
all_files_missed_current_tokens = []
all_files_missed_tokens_by_start = []

# Load the spaCy language model for Spanish
nlp = spacy.load("es_core_news_sm")


def tokenize_file(filename=None):
    """This function tokenizes the whole file and set the initial NER tags with 0 tags for all tokens

    Args:
        filename (string): e.g., es-S0210-56912007000900007-3.txt

    Returns:
        tokens: list of tuples (token_text, start_index, end_index, flag)
        tokens_by_start: list of tuples (start_index) to search for a token and replace its values when needed
        tokens_by_end: list of tuples (end_index) to search for a token and replace its values when needed
        ner_tags: list of NER tags that correspond to each token in tokens list
    """
    if filename is not None:
        # Open the file in read mode
        with open(filename, "r", encoding="utf-8") as file:
            # Read the entire contents of the file
            text = file.read()

        doc = nlp(text)
        tokens = []
        tokens_by_start = []
        tokens_by_end = []
        ner_tags = []
        for token in doc:
            if token.text != " ":
                flag = "O"
                tag = 0
                start_index = token.idx
                end_index = token.idx + len(token.text)
                tokens.append((token.text, start_index, end_index, flag))
                tokens_by_start.append((start_index))
                tokens_by_end.append((end_index))
                ner_tags.append(tag)

    return tokens, tokens_by_start, tokens_by_end, ner_tags


def tokenize_text(
    filename=None,
    tokens=None,
    tokens_by_start=None,
    tokens_by_end=None,
    ner_tags=None,
    text=None,
    start_span=None,
):
    """This function tokenizes the text (text column) and replaces the initial NER tags with 1 or 2 tags

    Args:
        filename (string): e.g., es-S0210-56912007000900007-3.txt
        tokens (list): list of tuples (token_text, start_index, end_index, flag)
        tokens_by_start (list): list of tuples (start_index) to search for a token and replace its values when needed
        tokens_by_end (list): list of tuples (end_index) to search for a token and replace its values when needed
        ner_tags (list): list of NER tags that correspond to each token in tokens list
        text (string): positive SINTOMA text (text column)
        start_span (int): start index of the SINTOMA text in the whole file

    Returns:
        tokens: list of tuples (token_text, start_index, end_index, flag)
        ner_tags: list of NER tags that correspond to each token in tokens list
        missed_texts: list of texts that have a problem
        missed_positive_tokens: list of positive tokens that have a problem
        missed_tokens: list of tokens that have a problem
        missed_current_tokens: list of current tokens that could not be found by searching for their start or end index
        missed_tokens_by_start: list of tokens by start index that have a problem
    """
    missed_texts = []
    missed_positive_tokens = []
    missed_tokens = []
    missed_current_tokens = []
    missed_tokens_by_start = []
    if text is not None:
        doc = nlp(text)
        positive_tokens = []
        for token in doc:
            if token.text != " ":
                start_index = token.idx + start_span
                end_index = start_index + len(token.text)
                current_token = (token.text, start_index, end_index)
                positive_tokens.append(current_token)

        for i in range(len(positive_tokens)):
            current_token = positive_tokens[i]
            if i == 0:
                flag = "B-SINTOMA"
                tag = 1
            else:
                flag = "I-SINTOMA"
                tag = 2

            try:
                # Find the index of the current token in the tokens_by_start list
                index = tokens_by_start.index((current_token[1]))
                # Replace the initial flag and NER tag with the new one
                tokens[index] = (
                    tokens[index][0],
                    tokens[index][1],
                    tokens[index][2],
                    flag,
                )
                ner_tags[index] = tag
            except:
                try:
                    # Check if we can find it by the end index
                    index = tokens_by_end.index((current_token[2]))
                    # Replace the initial flag and NER tag with the new one
                    tokens[index] = (
                        tokens[index][0],
                        tokens[index][1],
                        tokens[index][2],
                        flag,
                    )
                    ner_tags[index] = tag
                except:
                    print(
                        "Token index mismatch, so this file ", filename, "has a problem"
                    )
                    missed_texts.append(text)
                    missed_positive_tokens.append(positive_tokens)
                    missed_tokens.append(tokens)
                    missed_current_tokens.append(current_token)
                    missed_tokens_by_start.append(tokens_by_start)

    return (
        tokens,
        ner_tags,
        missed_texts,
        missed_positive_tokens,
        missed_tokens,
        missed_current_tokens,
        missed_tokens_by_start,
    )


def preprocess():
    # Get all unique file names from the dataset
    unique_filenames = df["filename"].unique()
    for filename in unique_filenames:
        # Get subest for the dataset (only records that belong to that filename)
        filtered_df = df[df["filename"] == filename]

        # Tokenize the text file and initialize the flags and the NER tag for each token
        tokens, tokens_by_start, tokens_by_end, ner_tags = tokenize_file(
            filename=f"{TXT_FILES_PATH}{filename}.txt"
        )

        for _, row in filtered_df.iterrows():
            ann_id = row["ann_id"]
            start_span = row["start_span"]
            end_span = row["end_span"]
            text = row["text"]  # SINTOMA text

            # Tokenize only SINTOMA texts and replace the initial flags and NER tags of these tokens
            (
                tokens,
                ner_tags,
                missed_texts,
                missed_positive_tokens,
                missed_tokens,
                missed_current_tokens,
                missed_tokens_by_start,
            ) = tokenize_text(
                filename=f"{TXT_FILES_PATH}{filename}.txt",
                tokens=tokens,
                tokens_by_start=tokens_by_start,
                tokens_by_end=tokens_by_end,
                ner_tags=ner_tags,
                text=text,
                start_span=start_span,
            )

            # Keep track of faulty files
            if len(missed_texts) > 0:
                all_files_missed_filenames.append(filename)
                all_files_missed_texts.append(missed_texts)
                all_files_missed_positive_tokens.append(missed_positive_tokens)
                all_files_missed_tokens.append(missed_tokens)
                all_files_missed_current_tokens.append(missed_current_tokens)
                all_files_missed_tokens_by_start.append(missed_tokens_by_start)

        # Each file has a unique ID and a list of tokens with their flags and NER tags
        global file_id
        file_id += 1
        filenames.append(filename)
        id_sequence.append(file_id)
        token_sequence.append([t[0] for t in tokens])
        ner_tags_sequence.append(ner_tags)
        start_end_sequence.append([[t[1], t[2]] for t in tokens])

    # Store faulty files
    pd.DataFrame(
        {
            "filename": all_files_missed_filenames,
            "text": all_files_missed_texts,
            "positive_tokens": all_files_missed_positive_tokens,
            "tokens": all_files_missed_tokens,
            "current_token": all_files_missed_current_tokens,
            "tokens_by_start": all_files_missed_tokens_by_start,
        }
    ).to_csv(
        f"{DATA_PATH}tsv/missing_tokens.tsv",
        sep="\t",
        index=False,
    )


def create_ner_dict(id_sequence, tokens, ner_tags):
    """Create a dictionary from the lists

    Args:
        id_sequence (list): unique id for each file
        tokens (list): tokens of each file
        ner_tags (list): NER tags of each file

    Returns:
        dictionary: dictionary with the data
    """
    return {
        "id": id_sequence,
        "tokens": tokens,
        "ner_tags": ner_tags,
    }


def create_dict(filenames, id_sequence, tokens, ner_tags, start_end_spans):
    """Create a dictionary from the lists

    Args:
        id_sequence (list): unique id for each file
        tokens (list): tokens of each file
        ner_tags (list): NER tags of each file
        start_end_spans (list): start and end spans of each token

    Returns:
        dictionary: dictionary with the data
    """
    return {
        "filename": filenames,
        "id": id_sequence,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "start_end_spans": start_end_spans,
    }


def create_hf_dataset(train_dict, validation_dict):
    """Create the Hugging Face dataset with splits

    Args:
        train_dict (dictionary): dictionary with the training data
        validation_dict (dictionary): dictionary with the validation data
        test_dict (dictionary): dictionary with the test data
    """
    features = Features(
        {
            "id": Value("int64"),
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=["O", "B-SINTOMA", "I-SINTOMA"])),
        }
    )

    # Create the Hugging Face dataset with splits
    train_SympTEMIST_dataset = Dataset.from_dict(train_dict, features)
    validation_SympTEMIST_dataset = Dataset.from_dict(validation_dict, features)
    SympTEMIST_splits = DatasetDict(
        {
            "train": train_SympTEMIST_dataset,
            "validation": validation_SympTEMIST_dataset,
        }
    )

    print(SympTEMIST_splits)
    # Save the dataset to disk
    SympTEMIST_splits.save_to_disk(f"{DATA_PATH}processed_SympTEMIST")


def create_csv_files(csv_file_path, dict):
    """Save data in CSV format

    Args:
        csv_file_path (string): file path to save the CSV file
        dict (dictionary): dictionary with the data to be saved
    """
    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame(dict)
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)


# Define main function
if __name__ == "__main__":
    # Preprocess the data for NER task
    preprocess()

    # Calculate the sizes for each split
    total_size = len(id_sequence)
    # Adding 7 becuase I noticed that files should start from T1
    # TODO: 7 needs to be dynamic
    train_size = int(0.95 * total_size) + 7

    # Create dictionaries from the lists
    train_ner_dict = create_ner_dict(
        id_sequence[:train_size],
        token_sequence[:train_size],
        ner_tags_sequence[:train_size],
    )
    validation_ner_dict = create_ner_dict(
        id_sequence[train_size:],
        token_sequence[train_size:],
        ner_tags_sequence[train_size:],
    )

    # Create and save the Hugging Face dataset
    create_hf_dataset(train_ner_dict, validation_ner_dict)

    train_dict = create_dict(
        filenames[:train_size],
        id_sequence[:train_size],
        token_sequence[:train_size],
        ner_tags_sequence[:train_size],
        start_end_sequence[:train_size],
    )

    validation_dict = create_dict(
        filenames[train_size:],
        id_sequence[train_size:],
        token_sequence[train_size:],
        ner_tags_sequence[train_size:],
        start_end_sequence[train_size:],
    )

    # Save data in CSV format
    create_csv_files(f"{DATA_PATH}train_SympTEMIST_dataset.csv", train_dict)
    create_csv_files(f"{DATA_PATH}validation_SympTEMIST_dataset.csv", validation_dict)

    # TSV file for evaluation
    eval_df = df[df["filename"].isin(filenames[train_size:])]
    eval_df.to_csv(f"{DATA_PATH}eval_dataset.tsv", index=False, sep="\t")

    print("Data Preprocessing is Completed!")
