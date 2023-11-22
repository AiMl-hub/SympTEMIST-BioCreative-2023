"""
MedProcNER evaluation library main script.
Partially based on the DisTEMIST and MEDDOPLACE evaluation scripts.
@author: salva
"""

import sys
import os

import pandas as pd

from datetime import datetime
from argparse import ArgumentParser

import utils

def main(argv=None):
    """
    Parse options and call the appropriate evaluation scripts
    """
    # Parse options
    parser = ArgumentParser()
    parser.add_argument("-r", "--reference", dest="reference",
                      help=".TSV file with Gold Standard or reference annotations", required=True)
    parser.add_argument("-p", "--prediction", dest="prediction",
                      help=".TSV file with your predictions", required=True)
    parser.add_argument("-t", "--task", dest="task", choices=['ner', 'norm', 'index'],
                      help="Task that you want to evaluate (ner, norm or index)", required=True)
    parser.add_argument("-o", "--output", dest="output",
                      help="Path to save the scoring results", required=True)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                      help="Set to True to print the results for each individual file instead of just the final score")
    args = parser.parse_args(argv)

    # Set output file name
    timedate = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(args.output, 'medprocner_results_{}_{}.txt'.format(args.task, timedate))

    # Read gold_standard and predictions
    print("Reading reference and prediction .tsv files")
    df_gs = pd.read_csv(args.reference, sep="\t")
    df_preds = pd.read_csv(args.prediction, sep="\t")
    if args.task in ['ner', 'norm']:
        df_preds = df_preds.drop_duplicates(
            subset=["filename", "label", "start_span", "end_span"]).reset_index(drop=True)  # Remove any duplicate predictions

    if args.task == "ner":
        calculate_ner(df_gs, df_preds, out_file, args.verbose)
    elif args.task == "norm":
        calculate_norm(df_gs, df_preds, out_file, args.verbose)
    elif args.task == "index":
        calculate_index(df_gs, df_preds, out_file, args.verbose)
    else:
        print('Please choose a valid task (ner, norm, index)')


def calculate_ner(df_gs, df_preds, output_path, verbose=False):
    print("Computing evaluation scores for Task 1 (ner)")
    # Group annotations by filename
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[[
        "filename", 'start_span', 'end_span', "text",  "label"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text", "label"]].values.tolist()).to_list()
    scores = utils.calculate_fscore(list_gs_per_doc, list_preds_per_doc, 'ner')
    utils.write_results('ner', scores, output_path, verbose)

def calculate_norm(df_gs, df_preds, output_path, verbose=False):
    print("Computing evaluation scores for Task 2 (norm)")
    # Group annotations by filename
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[[
        "filename", 'start_span', 'end_span', "text", "label", "code"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text", "label", "code"]].values.tolist()).to_list()
    scores = utils.calculate_fscore(list_gs_per_doc, list_preds_per_doc, 'norm')
    utils.write_results('norm', scores, output_path, verbose)

def calculate_index(df_gs, df_preds, output_path, verbose=False):
    print("Computing evaluation scores for Task 3 (index)")
    # Group annotations by filename
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[[
        "filename", "codes"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", "code"]].values.tolist()).to_list()
    scores = utils.calculate_fscore(list_gs_per_doc, list_preds_per_doc, 'index')
    utils.write_results('index', scores, output_path, verbose)

if __name__ == "__main__":
    main()