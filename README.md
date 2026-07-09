# Implementation Code of "Rethinking Explainable Disease Prediction: Synergizing Accuracy and Reliability via Reflective Cognitive Architecture"

This repository contains the official implementation of the paper "Rethinking Explainable Disease Prediction: Synergizing Accuracy and Reliability via Reflective Cognitive Architecture"

## Environment Setup

Clone this repository:
```bash
git clone https://github.com/ssssszj/RCA.git
cd RCA
```
Create a virtual environment (optional but recommended):
```bash
conda create -n rca_env python=3.9
conda activate rca_env
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
The paper evaluates RCA on five datasets. Three public datasets can be downloaded from the following sources:

- Diabetes: https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes/data

- Heart Disease: https://www.kaggle.com/datasets/oktayrdeki/heart-disease

- Cardiovascular Disease: https://www.kaggle.com/datasets/alamshihab075/heart-failure-diagnosis-data-for-machine-learning/data

The remaining two datasets, CRT and CRT_ex, are proprietary clinical cohorts used under institutional ethics and data-governance approval. They are not redistributed in this repository. For these datasets, the repository provides implementation code and prompt templates only.

Before running RCA, transform each tabular dataset into the text and label files expected by `main.py`. Each patient record should be converted into an unstructured feature narrative, and labels should be stored in the corresponding label file.

## Running Experiments
Set your LLM API key before running experiments. For details, see `utils/llm.py`.
```bash
python main.py 
--label_dir "Diabetes_data/labels/rawdata.txt" #path to labels
--feature_dir "Diabetes_data/texts/rawdata"  
#path to texts
--save_dir "results/Diabetes/rawdata"
#path to save results. If you want to utilize data distribution, distribution.json also should be put here
--num_epochs 30 
#epochs of training
--group_size 25 
#capacity of error groups
--train 
#whether train
--test 
#whether test
--load_ckpt 1
#Resume from breakpoint
```
Then you can find results in "save_dir".
