# ToMi Evaluation

This directory contains:
* scripts for processing and evaluating language models on the ToMi dataset (the models tested in these experiments are GPT-3, ChatGPT, GPT-4, o1, and Llama 3.1).
* documents containing some examples of issues in the data
* cleaned version of the original dataset

## Scripts

### 1. `preprocess_ToMi.py`
A utility script that converts the raw ToMi dataset into a JSONL format for easier processing.

**Usage:**
```bash
python preprocess_ToMi.py --input INPUT_PATH --output OUTPUT_PATH
```

### 2. `predict_ToMi-modified_cleaned.py`
Runs predictions on the ToMi dataset using a sequential approach (no conversation history).

**Usage:**
```bash
python predict_ToMi-modified_cleaned.py --model MODEL_NAME --dataset DATASET_NAME --predict [--max_tokens MAX_TOKENS] [--data_dir DATA_DIR] [--results_dir RESULTS_DIR] [--preprocess]
```

### 3. `predict_ToMi-conversation-history_cleaned.py`
Runs predictions on the ToMi dataset using conversation history for additional context.

**Usage:**
```bash
python predict_ToMi-conversation-history_cleaned.py --model MODEL_NAME --dataset DATASET_NAME --predict [--max_tokens MAX_TOKENS] [--data_dir DATA_DIR] [--results_dir RESULTS_DIR] [--preprocess]
```

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
  - huggingface-hub
  - tqdm

## Documents
This folder contains two documents:
* **Data Issues Analysis:** here we present some examples of the different ToMi data issues discussed in the paper
* **Human Evaluation Guidelines**: here we present guidelines for human to annotate the ToMi data items

## Cleaned Data
We started the experiment generating 100 stories (600 questions) from the [original ToMi code](https://github.com/facebookresearch/ToMi). After human evaluation of the predicted results, we found 50 grouth truth issues and we removed those items from the dataset.