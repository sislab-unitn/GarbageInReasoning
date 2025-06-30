from sklearn.metrics import accuracy_score
import csv
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate SocialIQA accuracy and analyze errors')
    parser.add_argument('--model', type=str, required=True,
                      help='Model name used for predictions')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name used')
    parser.add_argument('--results_dir', type=str, default='cleaned_results',
                      help='Directory containing model results')
    parser.add_argument('--data_dir', type=str, default='data/cleaned_socialiqa',
                      help='Directory containing original labels (default: data/cleaned_socialiqa)')
    parser.add_argument('--output_dir', type=str, default='error_analysis',
                      help='Directory to save error analysis (default: error_analysis)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_path_model_responses = os.path.join(args.results_dir, args.model, f"{args.dataset}.txt")
    input_path_original_labels = os.path.join(args.data_dir, "dev-labels_cleaned.lst")
    siqa_json = os.path.join(args.data_dir, f"{args.dataset}.jsonl")
    output_csv = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_errors.csv")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(input_path_model_responses, "r") as file1:
        siqa_pred = file1.read().splitlines()
    with open(input_path_original_labels, "r") as file2:
        siqa_true = file2.read().splitlines()
    
    accuracy = accuracy_score(siqa_true, siqa_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    with open(siqa_json) as f:
        dataset = [json.loads(line) for line in f]
    
    data = []
    for i in range(len(siqa_true)):
        if siqa_true[i] != siqa_pred[i]:
            dataset_item = dataset[i]
            if 'want' in dataset_item['question']:
                category = 'want'
            elif 'feel' in dataset_item['question']:
                category = 'reaction'
            elif 'describe' in dataset_item['question']:
                category = 'description'
            elif 'Why' in dataset_item['question']:
                category = 'motivation'
            elif 'need' in dataset_item['question']:
                category = 'need'
            elif 'happen to' in dataset_item['question']:
                category = 'effect'
            print(f"Error at item {i+1}: Predicted {siqa_pred[i]}, True {siqa_true[i]}")
            data.append({
                'ID': i+1,
                'Dataset item': dataset_item,
                'Question type': category,
                f'{args.model} label': siqa_pred[i],
                'Original label': siqa_true[i]
            })
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'Dataset item', 'Question type', f'{args.model} label', 'Original label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\nError analysis saved to: {output_csv}")

if __name__ == "__main__":
    main()