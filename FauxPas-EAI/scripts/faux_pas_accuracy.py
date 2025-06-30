import os
import argparse
from pathlib import Path
from typing import List, Tuple
from FauxPasEvaluation import FauxPasEvaluation
from FauxPasEAIParser import FauxPasEAIParser

def load_predictions(file_path: str) -> List[str]:
    """
    Load model predictions from a file.
    
    Args:
        file_path: Path to the file containing model predictions
        
    Returns:
        List of prediction strings, one per line
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().splitlines()
    except FileNotFoundError:
        print(f"Error: Could not find prediction file at {file_path}")
        return []

def load_ground_truth(parser: FauxPasEAIParser) -> List[str]:
    """
    Load ground truth answers from the Faux Pas dataset.
    
    Args:
        parser: An initialized FauxPasEAIParser instance
        
    Returns:
        List of ground truth answers
    """
    ground_truth = []
    
    for story_id, test_data in parser.tests.items():
        print("**************************************")
        print(story_id)
        
        (story, question1, answer1, question2, answer2, 
         question3, answer3, question4, answer4) = test_data
        
        ground_truth.extend([answer1, answer2, answer3, answer4])
    
    return ground_truth

def evaluate_model(predictions: List[str], ground_truth: List[str]) -> Tuple[float, float]:
    """
    Evaluate model predictions against ground truth.
    
    Args:
        predictions: List of model predictions
        ground_truth: List of ground truth answers
        
    Returns:
        Tuple of (question_level_accuracy, story_level_accuracy)
    """
    print(f"Number of ground truth answers: {len(ground_truth)}")
    print(f"Number of predictions: {len(predictions)}")
    
    # Calculate question-level accuracy
    question_accuracy = FauxPasEvaluation.compare_lists_question_level(
        predictions, ground_truth
    )
    
    # Calculate story-level accuracy
    story_accuracy = FauxPasEvaluation.compare_lists_story_level(
        predictions, ground_truth
    )
    
    return question_accuracy, story_accuracy

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate language model performance on the Faux Pas task."
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="davinci-002",
        help="Model name to evaluate (e.g., 'davinci-002', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'o1-preview')"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing the results files (default: parent directory of the script)"
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    model_name = args.model
    print(f"Evaluating model: {model_name}")
    
    if args.results_dir:
        base_dir = Path(args.results_dir)
    else:
        base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    
    input_path = base_dir / f"fauxpas-{model_name}-results.txt"
    print(f"Looking for results file at: {input_path}")
    
    predictions = load_predictions(str(input_path))
    if not predictions:
        print(f"No predictions found for model {model_name}. Exiting.")
        return
        
    parser = FauxPasEAIParser()
    ground_truth = load_ground_truth(parser)
    
    question_accuracy, story_accuracy = evaluate_model(predictions, ground_truth)
    
    print("\nResults:")
    print(f"Model: {model_name}")
    print(f"Accuracy (question level): {question_accuracy:.4f}")
    print(f"Accuracy (story level): {story_accuracy:.4f}")

if __name__ == "__main__":
    main()