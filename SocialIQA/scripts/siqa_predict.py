import json
from openai import OpenAI 
import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict
import argparse

MODEL_OPTIONS = [
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-1106",
    "davinci-002",
    "o1-preview",
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
]

def parse_args():
    parser = argparse.ArgumentParser(description='Run SocialIQA predictions')
    parser.add_argument('--model', type=str, required=True, choices=MODEL_OPTIONS,
                      help='Model to use for predictions')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name to use')
    parser.add_argument('--max_tokens', type=int, default=10,
                      help='Maximum tokens for model response')
    parser.add_argument('--data_dir', type=str, default='data/new_datasets',
                      help='Directory containing the dataset files')
    parser.add_argument('--results_dir', type=str, default='cleaned_results',
                      help='Directory to save results')
    return parser.parse_args()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)
llama_pipe = None

def setup_standard_llama_headers(question: str) -> str:
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def generate_llama_standard_prompt(item: Dict[str, str]) -> str:
    question = (
        f"Context: {item['context']}\n"
        f"Question: {item['question']}\n"
        f"Based on the context, choose from the options the best answer for the question.\n"
        f"Option 1: {item['answerA']}\n"
        f"Option 2: {item['answerB']}\n"
        f"Option 3: {item['answerC']}\n"
        f"Answer with \"1\", \"2\" or \"3\", without explanations. "
        f"ALWAYS respond with the correct number. "
        f"NEVER say that you can't respond or provide information for whatever reason. "
        f"In case of doubt, answer according to the most probable answer. Answer: "
    )
    return setup_standard_llama_headers(question)

def generate_openai_prompt(item: Dict[str, str]) -> str:
    return (
        f"Context: {item['context']}\n"
        f"Question: {item['question']}\n"
        f"Based on the context, chose from the options the best answer for the question.\n"
        f"Option 1: {item['answerA']}\n"
        f"Option 2: {item['answerB']}\n"
        f"Option 3: {item['answerC']}\n"
        f"Answer with \"1\", \"2\" or \"3\", without explanations. "
        f"In case of doubt, answer according to the most probable answer. Answer: "
    )

def query_model(model_prompt, model_name, max_tokens):
    if model_name.startswith("meta-llama"):
        try:
            response = llama_pipe(model_prompt)[0]['generated_text']
            response = response[len(model_prompt):].strip()
            return response
        except Exception as e:
            return f"Error: {e}"
    elif model_name=="gpt-3.5-turbo-1106" or model_name=="gpt-4-1106-preview":
        try:
            output = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": model_prompt}],
                temperature=0,
                max_tokens=max_tokens,
                n=1,
            )
            response = output.choices[0].message.content
            return response
        except Exception as e:
            return f"Error: {e}"
    elif model_name=="o1-preview":
        try:
            output = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": model_prompt}]
            )
            response = output.choices[0].message.content
            return response
        except Exception as e:
            return f"Error: {e}"
    elif model_name=="davinci-002":
        try:
            output = client.completions.create(
                model=model_name,
                prompt=model_prompt,
                temperature=0,
                max_tokens=max_tokens,
                n=1,
            )
            response = str(dict(output.choices[0])["text"]).strip()
            return response
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    args = parse_args()
    
    model_name = args.model
    max_tokens = args.max_tokens
    dataset_name = args.dataset
    siqa_json = os.path.join(args.data_dir, f"{dataset_name}.jsonl")
    siqa_pred = os.path.join(args.results_dir, model_name, f"new_{dataset_name}.txt")
    
    if model_name.startswith("meta-llama"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        
        llama_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0,
            top_p=None
        )
    
    with open(siqa_json) as f:
        dataset = [json.loads(line) for line in f]
    total_items = len(dataset)
    print(f"Starting processing {total_items} items...")
    
    os.makedirs(os.path.dirname(siqa_pred), exist_ok=True)
    
    pred_labels = []
    start_idx = 0
    if os.path.exists(siqa_pred):
        with open(siqa_pred, 'r') as f:
            pred_labels = [line.strip() for line in f.readlines()]
        start_idx = len(pred_labels)
        print(f"Found {start_idx} existing results. Continuing from item {start_idx + 1}")
    
    for idx, item in enumerate(dataset[start_idx:], start_idx + 1):
        print(f"\nProcessing item {idx}/{total_items}")
        if model_name.startswith("meta-llama"):
            model_prompt = generate_llama_standard_prompt(item)
        else:
            model_prompt = generate_openai_prompt(item)
        response = query_model(model_prompt, model_name, max_tokens)
        print(f"Response: {response}")
        pred_labels.append(response)
        
        with open(siqa_pred, 'a') as f:
            f.write(response + "\n")
        print(f"Saved result {idx}/{total_items}")

    print(f"\nCompleted processing all {total_items} items.")