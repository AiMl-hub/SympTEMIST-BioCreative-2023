# Fine-tuned Large Language Models for Symptom Recognition from Spanish Clinical Text
Mai A. Shaaban, Abbas Akkasi, Adnan Khan, Majid Komeili, Mohammad Yaqub

**Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE**

**School of  Computer Science,  Carleton University, Ottawa, CA**


[![Static Badge](https://img.shields.io/badge/GitHub-%20AiMl-fuchsia?link=https%3A%2F%2Fgithub.com%2FAiMl-hub)](https://github.com/AiMl-hub)
[![Static Badge](https://img.shields.io/badge/Paper-Link-yellowgreen?link=https%3A%2F%2Fzenodo.org%2Frecords%2F10104139)](https://zenodo.org/records/10104139)
[![python](https://img.shields.io/badge/Python-3.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-1.7.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

## Abstract

The accurate recognition of symptoms in clinical reports is significantly important in the fields of healthcare and biomedical natural language processing. These entities serve as essential building blocks for clinical information extraction, enabling retrieval of critical medical insights from vast amounts of textual data. Furthermore, the ability to identify and categorize these entities is fundamental for developing advanced clinical decision support systems, aiding healthcare professionals in diagnosis and treatment planning. In this study, we participated in SympTEMIST – a shared task on detection of symptoms, signs and findings in Spanish medical documents. We combine a set of large language models finetuned with the data released by the task's organizers.


This article is part of the Proceedings of the BioCreative VIII Challenge and Workshop: Curation and Evaluation in the era of Generative Models.

# SympTEMIST: Symptoms, Signs and Findings Entity Recognition and Linking Shared Task @ BioCreative 2023
The [SympTEMIST Track](https://temu.bsc.es/symptemist/) is organized by the Barcelona Supercomputing Center’s NLP for Biomedical Information Analysis group and promoted by Spanish and European projects such as DataTools4Heart, AI4HF, BARITONE and AI4ProfHealth.
## Subtask 1: SymptomNER
In case of this main subtask, participants will be asked to automatically detect mention spans of symptoms (including also signs and findings) from clinical reports written in Spanish. Using the SympTEMIST corpus as training data (manually labelled mentions), they must create systems that are able to return the start and end position (character offsets) of all symptom entities mentioned in the text. The main evaluation metrics for this task will be precision, recall and f-score.

Note that we also provide access to additional other annotations for the documents used as training data beyond symptoms, which could be exploited further to improve the systems, namely the annotation of a) diseases (Distemist corpus), b) clinical procedures (MedProcNER corpus) and c) chemical compounds, drugs and genes/proteins (PharmaCoNER corpus). Links to these other resources will be posed at the Symptemist corpus zenodo webpage.

## How to fine-tune ️▶️

1) First, clone the project.

2) Set up a Python virtual environment and activate it

   `conda create -n symp_env python=3.8`
   
   `conda activate symp_env`

3) Install all the required python packages in the virtual environment running the following line from the project main folder:

   `pip install -r requirements.txt`

   For preprocessing the data (optional), you will also need to install the following packages:

   `python -m spacy download es_core_news_sm`

4) Finally, run the following command on the project main folder: `bash symp_ner.sh [MODEL_NAME] [DATASET_NAME] [SEED] [OFFLINE] [EPOCHS]`, where

  - `[MODEL_NAME]`: HuggingFace' model name of the pretrained model you want to use.
  - `[DATASET_NAME]`: HuggingFace' dataset name of the NER dataset to use OR dataset path on local disk.
  - `[SEED]`: the seed you want to use. This allows to reproduce the same results.
  - `[OFFLINE]`: boolean variable to determine whether to load datasets from HuggingFace or local disk.
   - `[EPOCHS]`: number of epochs to train the model.

The `symp_ner.sh` script fine-tune a pretrained language model for the NER task applying a linear classification head. By default, the fine-tuning run for 10 epochs with an evaluation on the development set at every epoch. The model achieving the best performance on the development set is selected as the final model and evaluated on the test set. The best trained model is store in a output path of the type `./output/model-$model_name/dataset-$dataset_name/seed-$seed` along with the checkpoints folders and the tensorboard data (inside the `tb` directory). 

For example, to fine-tune the [bsc-bio-es](https://huggingface.co/PlanTL-GOB-ES/bsc-bio-es) language model on the SympTEMIST dataset, run the command: 

```
bash symp_ner.sh PlanTL-GOB-ES/bsc-bio-es data/subtask1-ner/processed_SympTEMIST 42 True 70
```
Other models can be:
- PlanTL-GOB-ES/bsc-bio-ehr-es
- xlm-roberta-large
- xlm-roberta-base
- IIC/XLM-R_Galen
- intfloat/multilingual-e5-large

## How to evaluate ️▶️
```
cd medprocner_evaluation_library-main
```

```
python3 medprocner_evaluation.py -r ../data/subtask1-ner/eval_dataset.tsv -p ../results/model-bsc-bio-ehr-es/val.tsv -t ner -o scores/
```

```eval_dataset.tsv``` is the ground truth and ```val.tsv``` is the predictions file.

Each TSV file has the following format:
- filename: Name of the file from which the procedure mention has been extracted. (provided by the organizers)
- label: In our case it will always be SINTOMA. (prediction of your system)
- start_span: Character number where the detected mention starts. (prediction of your system)
- end_span: Character number where the detected mention ends. (prediction of your system)
- text: Mention extracted from text. (prediction of your system)


## Acknowledgement
The fine-tuning code is based on the work of [lm-biomedical-clinical-es](https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es). All credit goes to them.
