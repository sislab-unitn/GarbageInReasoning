import time
import openai
import json
import tqdm
from sklearn.metrics import accuracy_score
import re
import csv
import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

MODELS_IDS = {
    "llama-3.1-8B-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

MODEL_OPTIONS = ['llama-3.1-8B-instruct', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'davinci-002', 'o1-preview']

def joint_accuracy(gold, predictions):
    joint_results = []
    gold_joint = []
    predictions_joint = []
    for i, item in enumerate(predictions):
        if item == gold[i]:
            joint_results.append(True)
        else:
            joint_results.append(False)
        if (i + 1) % 6 == 0:
            prediction = all(element == joint_results[0] for element in joint_results)
            predictions_joint.append(prediction)
            gold_joint.append(True)
            joint_results = []
    accuracy_joint = accuracy_score(gold_joint, predictions_joint)
    print(f"Joint Accuracy: {accuracy_joint:.3f}")


def remove_whitespaces_and_numbers(input_string):
    # Remove all whitespaces and digits using regular expressions for a more fair accuracy results
    return re.sub(r'[0-9\s\.\,\!\?\"\'\-]', '', input_string)

def word_before_dot(input_string):
    # Use a regular expression to find a word followed by a dot
    match = re.search(r'\b(\w+)[\.\n]', input_string)
    if match:
        return match.group(1)
    return None

def average_accuracy(args): #with csv report of mislabeled
    gold = []
    predictions = []
    mislabeled_data = []
    file1 = open(f"./{args.data_dir}/test.trace", "r")
    question_types = file1.read().splitlines()
    file1.close()
    with open(f'./{args.results_dir}/{args.dataset}-{args.model}-results_conversation-history.txt') as f_in:
        for i, line in enumerate(tqdm.tqdm(f_in)):
            fields = json.loads(line)
            gold_label = remove_whitespaces_and_numbers(str(fields['label']))
            predicted_label = remove_whitespaces_and_numbers(str(fields['prediction']))
            if args.model == "davinci-002":
                predicted_label = word_before_dot(str(fields['prediction']))
            if gold_label != predicted_label:
                question_type = question_types[i].split(",")[-2:]
                fields["question type"] = question_type[0]
                fields["belief"] = question_type[1]
                mislabeled_data.append(fields)
            gold.append(gold_label)
            predictions.append(predicted_label)
    accuracy = accuracy_score(gold, predictions)
    print(f"Average Accuracy: {accuracy:.3f}")
    return gold, predictions, mislabeled_data



def predict(args, pipe=None):
    conversation_history = []
    with open(f'./{args.results_dir}/{args.dataset}-{args.model}-results_conversation-history.txt', "a") as f_out:
        with open(f"./{args.data_dir}/{args.dataset}.txt") as f_in:
            for i, line in enumerate(tqdm.tqdm(f_in)):
                if i >= 600:  # Stop after 600 lines for 600 questions (100 stories)
                    break
                fields = json.loads(line)
                if i % 6 == 0:
                    conversation_history = []
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that answers questions with a single word, no numbers, articles or explanations."}
                ]
                
                for msg in conversation_history:
                    messages.append(msg)
                
                current_prompt = fields['context'] + ' ' + fields['question'] + ' Answer with one word only, without explanations.'
                messages.append({"role": "user", "content": current_prompt})
                
                if args.model == "llama-3.1-8B-instruct":
                    fields['prediction'] = llama_finalanswer_request(messages, i, pipe)
                else:
                    fields['prediction'] = open_ai_finalanswer_request(messages, i, 0, args.model)
                    
                print(f"Prediction {i}: {fields['prediction']}")
                
                fields['conversation history'] = json.dumps(conversation_history)
                conversation_history.append({"role": "user", "content": current_prompt})
                conversation_history.append({"role": "assistant", "content": fields['prediction']})
                
                f_out.write(json.dumps(fields) + "\n")

def setup_llama_pipeline(
    model_name: str, 
    device: str, 
    max_tokens: int = 10, 
    do_sample: bool = False, 
    temperature: float = None, 
    top_p: float = None
    ) -> pipeline:
    
    print(f"Loading {model_name} model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    llama_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    return llama_pipeline

def llama_finalanswer_request(messages, i, llama_pipeline):
    try:
        outputs = llama_pipeline(
            messages,
            return_full_text=False
        )
        
        if not outputs or len(outputs) == 0:
            print(f"Warning: No output generated for prompt {i}")
            return "none"
            
        prediction = outputs[0]['generated_text'].strip().lower()
        prediction = prediction.rstrip('.')
            
        return prediction
    except Exception as e:
        print(f"Error: {str(e)}")
        return "none"

def preprocess_tomi(input_path, args):
    with open(f"./{args.data_dir}/{args.dataset}.txt", "a") as f_out:
        with open(input_path) as input_file:
            sample = dict()
            sample["context"] = ""
            for line in enumerate(input_file):
                print(line)
                if "?" in line[1]:
                    new_line = line[1].split("\t")
                    sample["context"] = sample["context"].strip()
                    sample["question"] = new_line[0].strip()
                    sample['label'] = new_line[1].strip() + " " + new_line[2].strip()
                    f_out.write(json.dumps(sample) + "\n")
                    sample = dict()
                    sample["context"] = ""
                else:
                    sample["context"] += line[1].strip() + "\n"


def open_ai_finalanswer_request(prompt, i, counter, model_name):
    if model_name=="gpt-3.5-turbo-1106" or model_name=="gpt-4-1106-preview":
            try:
                response = openai.chat.completions.create(
                        model=model_name,
                        temperature=0,
                        max_tokens=10,
                        messages=prompt
                    )
                prediction = response.choices[0].message.content.strip().lower()
                return prediction
            except Exception as e:
                prediction = f"Error: {str(e)}"
    elif model_name=="o1-preview":
            try:
                response = openai.chat.completions.create(
                model="o1-preview",
                #max_completion_tokens=max_tokens,
                messages=prompt,
                )
                prediction = response.choices[0].message.content.strip().lower()
                return prediction
            except Exception as e:
                prediction = f"Error: {str(e)}"
    elif model_name=="davinci-002":
        try:
            response = openai.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=0,
                max_tokens=30,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            prediction = str(dict(response.choices[0])["text"]).strip()
            return prediction
        except:
            if counter < 3:
                time.sleep(10)
                return open_ai_finalanswer_request(prompt, i, counter + 1)
            else:
                print(prompt)
                print("continue from:" + str(i))
                exit()

def parse_args():
    parser = argparse.ArgumentParser(description='Run ToMi predictions with conversation history')
    parser.add_argument('--model', type=str, required=True, choices=MODEL_OPTIONS,
                      help='Model to use for predictions')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset set to use (e.g., test, train)')
    parser.add_argument('--max_tokens', type=int, default=10,
                      help='Maximum tokens for model response')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing the dataset files')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--preprocess', action='store_true',
                      help='Preprocess the dataset (use only once to avoid duplicating data)')
    parser.add_argument('--predict', action='store_true',
                      help='Run predictions on the dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    
    openai.api_key = OPENAI_API_KEY
    
    login(HF_TOKEN)
    
    input_path = f"./{args.data_dir}/{args.dataset}.txt"    #file of the input non-preprocessed data
    if args.preprocess:
        preprocess_tomi(input_path, args)
    
    llama_pipeline = setup_llama_pipeline(MODELS_IDS[args.model], "cuda", args.max_tokens)
    
    if args.predict:
        predict(args, llama_pipeline)
    
    gold, predictions, mislabeled_data = average_accuracy(args)
    with open(f'./{args.results_dir}/ToMi_{args.dataset}-{args.model}-mislabeled_conversation-history.csv', 'w', newline='') as csvfile:
        fieldnames = ['context', 'question', 'label', 'prediction', 'conversation history', 'question type', 'belief']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mislabeled_data)
        
    joint_accuracy(gold, predictions)

if __name__ == "__main__":
    main()
