"""
MedProcNER evaluation library evaluation and util functions.
Partially based on the DisTEMIST and MEDDOPLACE evaluation scripts.
@author: salva
"""
# logger --> []
# METRICS
import logging

# now we will Create and configure logger
logging.basicConfig(
    filename="../log_files/exp.log", format="%(asctime)s %(message)s", filemode="w"
)

# Let us Create an object
logger = logging.getLogger()

# Now we are going to Set the threshold of logger to DEBUG
logger.setLevel(logging.INFO)
# dummy_pred = []
# dummy_gold = []


def calculate_fscore(gold_standard, predictions, task):
    """
    Calculate micro-averaged precision, recall and f-score from two pandas dataframe
    Depending on the task, do some different pre-processing to the data
    """
    # Cumulative true positives, false positives, false negatives
    total_tp, total_fp, total_fn = 0, 0, 0
    # Dictionary to store files in gold and prediction data.
    gs_files = {}
    pred_files = {}

    for document in gold_standard:
        document_id = document[0][0]
        gs_files[document_id] = document
    for document in predictions:
        document_id = document[0][0]
        pred_files[document_id] = document

    # Dictionary to store scores
    scores = {}

    # Iterate through documents in the Gold Standard
    for document_id in gs_files.keys():
        doc_tp, doc_fp, doc_fn = 0, 0, 0
        gold_doc = gs_files[document_id]
        #  Check if there are predictions for the current document, default to empty document if false
        if document_id not in pred_files.keys():
            predicted_doc = []
        else:
            predicted_doc = pred_files[document_id]
        if task == "index":  # Separate codes
            gold_doc = list(set(gold_doc[0][1].split("+")))
            predicted_doc = (
                list(set(predicted_doc[0][1].split("+"))) if predicted_doc else []
            )
        # Iterate through a copy of our gold mentions
        for gold_annotation in gold_doc[:]:
            # Iterate through predictions looking for a match
            for prediction in predicted_doc[:]:
                # Separate possible composite normalizations
                if task == "norm":
                    separate_prediction = prediction[:-1] + [
                        code.rstrip() for code in sorted(str(prediction[-1]).split("+"))
                    ]  # Need to sort
                    separate_gold_annotation = gold_annotation[:-1] + [
                        code.rstrip() for code in str(gold_annotation[-1]).split("+")
                    ]
                    if set(separate_gold_annotation) == set(separate_prediction):
                        # Add a true positive
                        doc_tp += 1
                        # Remove elements from list to calculate later false positives and false negatives
                        predicted_doc.remove(prediction)
                        gold_doc.remove(gold_annotation)
                        break
                if set(gold_annotation) == set(prediction):
                    # Add a true positive
                    doc_tp += 1
                    # print("I am stopping at breakpoint1")
                    # breakpoint()
                    # Remove elements from list to calculate later false positives and false negatives
                    # dummy_pred.append(prediction)
                    # dummy_gold.append(gold_annotation)
                    predicted_doc.remove(prediction)
                    gold_doc.remove(gold_annotation)
                    break
        # Get the number of false positives and false negatives from the items remaining in our lists
        doc_fp += len(predicted_doc)
        doc_fn += len(gold_doc)
        # Calculate document score
        try:
            precision = doc_tp / (doc_tp + doc_fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = doc_tp / (doc_tp + doc_fn)
        except ZeroDivisionError:
            recall = 0
        if precision == 0 or recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)
            logger.info(
                f"The F1 score for  {gold_annotation[0]} in gold standard and {prediction[0]} in predictions is {f_score}"
            )
        # Add to dictionary
        scores[document_id] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f_score": round(f_score, 4),
        }
        # Update totals
        total_tp += doc_tp
        total_fn += doc_fn
        total_fp += doc_fp
        # print("I am stopping at breakpoint2")
        # breakpoint()

    # Now let's calculate the micro-averaged score using the cumulative TP, FP, FN
    try:
        precision = total_tp / (total_tp + total_fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = total_tp / (total_tp + total_fn)
    except ZeroDivisionError:
        recall = 0
    if precision == 0 or recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    scores["total"] = {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f_score": round(f_score, 4),
    }

    return scores


# HELPER
def write_results(task, scores, output_path, verbose):
    """
    Helper function to write the results for each of the tasks
    """
    headers_dict = {
        "ner": "MedProcNER Shared Task: Subtask 1 (Named Entity Recognition) Results",
        "norm": "MedProcNER Shared Task: Subtask 2 (Entity Linking) Results",
        "index": "MedProcNER Shared Task: Subtask 3 (Document Indexing) Results",
    }

    with open(output_path, "w") as f_out:
        # This looks super ugly, but if we keep the indentation it will also appear in the output file
        f_out.write(
            """-------------------------------------------------------------------
{}
-------------------------------------------------------------------
""".format(
                headers_dict[task]
            )
        )
        if verbose:
            for k in scores.keys():
                if k != "total":
                    f_out.write(
                        """-------------------------------------------------------------------
Results for document: {}
-------------------------------------------------------------------
Precision: {}
Recall: {}
F-score: {}
""".format(
                            k,
                            scores[k]["precision"],
                            scores[k]["recall"],
                            scores[k]["f_score"],
                        )
                    )

        f_out.write(
            """-------------------------------------------------------------------
Overall results:
-------------------------------------------------------------------
Micro-average precision: {}
Micro-average recall: {}
Micro-average F-score: {}
""".format(
                scores["total"]["precision"],
                scores["total"]["recall"],
                scores["total"]["f_score"],
            )
        )
    print("Written MedProcNER {} scores to {}".format(task, output_path))
