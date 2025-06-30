# FauxPas-EAI Evaluation

This directory contains:
* scripts for evaluating language models on the Faux Pas task (the models tested in these experiments are GPT-3, ChatGPT, GPT-4, o1, and Llama 3.1)
* documents containing some examples of issues in the data and in the evaluation approach
* cleaned version of the original dataset

## Scripts
### Main Testing Scripts
- `TestGPTFauxPasEAI-modified.py`: Modified version of the GPT testing script, from the [original FauxPas-EAI repository](https://github.com/natalieShapira/FauxPasEAI/blob/main/Code%20and%20Data/TestGPTFauxPasEAI.py), that evaluates OpenAI models on the Faux Pas task
- `TestGPTFauxPasEAI-conversation_history.py`: Version that includes conversation history in the evaluation
- `TestLlamaFauxPasEAI.py`: Script for testing Llama models on the Faux Pas task
- `TestLlamaFauxPasEAI-conversation_history.py`: Version that includes conversation history for Llama model testing

### Evaluation and Analysis
- `faux_pas_accuracy.py`: Calculates and reports accuracy metrics for model predictions
- `llm_as_a_judge.py`: Implements an LLM-based evaluation system for judging model responses

### Usage
To test a model on the Faux Pas task:

```bash
python TestGPTFauxPasEAI-modified.py  # For GPT models
# or
python TestLlamaFauxPasEAI.py  # For Llama
```

To test a model on the Faux Pas task, using the conversation history:
```bash
python TestGPTFauxPasEAI-conversation_history.py  # For GPT models
# or
python TestLlamaFauxPasEAI-conversation_history.py  # For Llama
```

To evaluate model performance:

```bash
python faux_pas_accuracy.py --model MODEL_NAME [--results-dir RESULTS_DIRECTORY]
```

### Requirements

- Python 3.x
- API keys (set in environment variables or .env file)
  - OpenAI API key (for GPT scripts)
  - HuggingFace API key (for Llama scripts)
- Required Python packages:
  - openai
  - python-dotenv
  - torch
  - transformers
- Download [FauxPasEAIParser.py](https://github.com/natalieShapira/FauxPasEAI/blob/main/Code%20and%20Data/FauxPasEAIParser.py) and [FauxPasEvaluation.py](https://github.com/natalieShapira/FauxPasEAI/blob/main/Code%20and%20Data/FauxPasEvaluation.py) files from the [original FauxPas-EAI repository](https://github.com/natalieShapira/FauxPasEAI/tree/main)

## Documents
This folder contains two documents:
* **Data and Evaluation Issues Analysis:** here we present some examples of FauxPas-EAI data and evaluation issues discussed in the paper
* **Human Evaluation Guidelines**: here we present guidelines for human to annotate the FauxPas-EAI data items

## Cleaned Data
The cleaned dataset is a modified version of the [original FauxPas-EAI benchmark](https://github.com/NatalieShapira/FauxPasEAI). We removed 4 stories from the original 44, in particular story *1f*, *5c*, *14c*, and *22f*, because of data issues.