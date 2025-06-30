# SocialIQA Evaluation

This directory contains:
* scripts for evaluating language models on the SocialIQA dataset (the models tested in these experiments are GPT-3, ChatGPT, GPT-4, o1, and Llama 3.1)
* documents containing some examples of issues in the data
* cleaned version of the original dataset

## Scripts

### 1. `siqa_predict.py`
Runs predictions on the SocialIQA dataset using various language models.

**Usage:**
```bash
python siqa_predict.py --model MODEL_NAME --dataset DATASET_NAME [--max_tokens MAX_TOKENS] [--data_dir DATA_DIR] [--results_dir RESULTS_DIR]
```

### 2. `siqa_accuracy.py`
Calculates accuracy scores and performs error analysis on model predictions.

**Usage:**
```bash
python siqa_accuracy.py --model MODEL_NAME --dataset DATASET_NAME [--results_dir RESULTS_DIR] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
```

### 3. `socialiqa_rephrasing.py`
Contains classes and methods to generate rephrased versions of SocialIQA items using language models.

### Requirements

- Python 3.x
- API keys (set in environment variables or .env file)
  - OpenAI API key
  - HuggingFace API key
- Required Python packages:
  - openai
  - python-dotenv
  - torch
  - transformers
  - scikit-learn

## Documents
This folder contains two documents:
* **Data Issues Analysis:** here we present some examples of the different SocialIQA data issues discussed in the paper
* **Human Evaluation Guidelines**: here we present guidelines for human to annotate the SocialIQA data items

## Cleaned Data
The cleaned dataset is a modified version of the [original SocialIQA development set](https://huggingface.co/datasets/allenai/social_i_qa). After our evaluation process, we removed all the items for which we found issues in the data.