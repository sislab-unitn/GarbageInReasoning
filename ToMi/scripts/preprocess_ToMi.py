import json
import os
import argparse

def preprocess_tomi(input_path: str, output_path: str) -> None:
    """
    Converts the ToMi dataset into a JSONL file.

    Args:
        input_path: Path to the input ToMi dataset file
        output_path: Path where the preprocessed JSONL file will be saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "a") as f_out:
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

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess ToMi dataset into JSONL format",)
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input ToMi dataset file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path where the preprocessed JSONL file will be saved"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    preprocess_tomi(args.input, args.output)