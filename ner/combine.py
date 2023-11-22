# cd medprocner_evaluation_library-main
# python3 medprocner_evaluation.py -r ../data/subtask1-ner/eval_dataset.tsv -p ../results/combined_val.tsv -t ner -o scores/

import pandas as pd

# Read the TSV files into dataframes
xlm_large_file = "./results/model-xlm-roberta-large/test.tsv"
xlm_base_file = "./results/model-xlm-roberta-base/test.tsv"
bio_file = "./results/model-bsc-bio-es/test.tsv"
ehr_file = "./results/model-bsc-bio-ehr-es/test.tsv"
e5_large_file = "./results/model-multilingual-e5-large/test.tsv"
e5_base_file = "./results/model-multilingual-e5-base/test.tsv"
# xlm_large_file = "./results/model-xlm-roberta-large/val.tsv"
# xlm_base_file = "./results/model-xlm-roberta-base/val.tsv"
# bio_file = "./results/model-bsc-bio-es/val.tsv"
# ehr_file = "./results/model-bsc-bio-ehr-es/val.tsv"
# e5_large_file = "./results/model-multilingual-e5-large/val.tsv"
# e5_base_file = "./results/model-multilingual-e5-base/val.tsv"

df_xlm_large = pd.read_csv(xlm_large_file, sep="\t")
df_xlm_base = pd.read_csv(xlm_base_file, sep="\t")
df_bio = pd.read_csv(bio_file, sep="\t")
df_ehr = pd.read_csv(ehr_file, sep="\t")
df_e5_large = pd.read_csv(e5_large_file, sep="\t")
df_e5_base = pd.read_csv(e5_base_file, sep="\t")


# # eval f1-score = 0.6352
df = pd.concat(
    [
        df_ehr,
        df_ehr,
        df_ehr,
        df_ehr,
        df_bio,
        df_bio,
        df_bio,
        df_xlm_large,
        df_xlm_large,
        df_xlm_base,
        df_e5_large,
        df_e5_base,
    ],
    axis=0,
)

# eval f1-score = 0.6364
# df = pd.concat(
#     [
#         df_ehr,
#         df_ehr,
#         df_ehr,
#         df_ehr,
#         df_bio,
#         df_bio,
#         df_bio,
#         df_xlm_large,
#         df_xlm_large,
#         df_xlm_base,
#         df_e5_base,
#     ],
#     axis=0,
# )

# Count number of occurances of each row
df = df.groupby(df.columns.tolist(), as_index=False).size()


# Majority vote by row level
combined_df = pd.DataFrame(columns=df.columns)

unique_filenames = df["filename"].unique()
for filename in unique_filenames:
    # Get subest for the dataset (only records that belong to that filename)
    filtered_df = df[df["filename"] == filename]
    unique_ids = filtered_df["ann_id"].unique()
    for ann_id in unique_ids:
        filtered_df_by_ann = filtered_df[filtered_df["ann_id"] == ann_id]
        filtered_df_by_ann = filtered_df_by_ann.reset_index()
        majority_voted_row = filtered_df_by_ann.iloc[
            [filtered_df_by_ann["size"].idxmax()]
        ]
        combined_df = pd.concat(
            [combined_df, majority_voted_row], axis=0, ignore_index=True
        )

combined_df = combined_df.drop(["size", "index"], axis=1)
combined_df.to_csv("./results/combined_test.tsv", index=False, sep="\t")
# combined_df.to_csv("./results/combined_val.tsv", index=False, sep="\t")
