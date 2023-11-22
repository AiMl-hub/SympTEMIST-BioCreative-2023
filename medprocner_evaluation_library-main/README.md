# MedProcNER Evaluation Library

This repository contains the official evaluation library for the [MedProcNER/ProcTEMIST Shared Task](https://temu.bsc.es/meddprocner).
MedProcNER is a shared task/challenge and set of resources for the detection and normalization of clinical procedures in medical documents in Spanish.
For more information about the task, data, evaluation metrics, ... please visit the task's website.


## Requirements

To use this scorer, you'll need to have Python 3 installed in your computer. Clone this repository, create a new virtual environment and then install the required packages:

```bash
git clone https://github.com/TeMU-BSC/medprocner_evaluation_library
cd medprocner_evaluation_library
python3 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
```

The MedProcNER task data is available on [Zenodo](https://doi.org/10.5281/zenodo.7817745). Keep in mind that the reference test set won't be uploaded until the task has finished.

## Usage Instructions

This program compares two .TSV files, with one being the reference file (i.e. Gold Standard data provided by the task organizers) and the other being the predictions or results file (i.e. the output of your system). Your .TSV file needs to have the following structure:

- For sub-task 1 (Named Entity Recognition): filename, label, start_span, end_span, text
- For sub-task 2 (Entity Linking): filename, label, start_span, end_span, text, code (multiple codes need to be separated with a '+' sign)
- For sub-task 3 (Indexing): filename, code (multiple codes need to be separated with a '+' sign)

Once you have your predictions file in the appropriate format and the reference data ready, you can run the library from your terminal using the following command:

```commandline
python3 medprocner_evaluation.py -r toy_data/medprocner_toy_task1_ref.tsv -p toy_data/medprocner_toy_task1_pred.tsv -t ner -o scores/
```

The output will be a .txt file saved in your desired location (`-o` option) with the following filename: medprocner_results_{task}_{timestamp}.txt

These are the possible arguments:

+ ```-r/--reference```: path to Gold Standard TSV file with the annotations
+ ```-p/--prediction```: path to predictions TSV file with the annotations
+ ```-o/--output```: path to save the scoring results file
+ ```-t/--task```: subtask name (```ner```, ```norm```, or ```index```).
+ ```-v/--verbose```: whether to include the evaluation of every individual document in the scoring results file


### Citation
This section will be updated soon with the citation info.
