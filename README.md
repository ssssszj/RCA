# Implementation Code of "Grounding AI Explanations in Eeperience: A Reflective Cognitive Architecture for Clinical Decision Support"

This repository contains the official implementation of the paper "Grounding AI Explanations in Eeperience: A Reflective Cognitive Architecture for Clinical Decision Support"

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
We used three datasets in the paper, two of which are open-sourced. Links are listed below:

Diabetes: https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes/data

HeartDisease: https://www.kaggle.com/datasets/oktayrdeki/heart-disease

You need to transform tabular data to unstructured texts and labels.

## Running Experiments
Now you can run experiment happily. Don't forget to set your llm api_key. For details, see utils/llm.py.
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
