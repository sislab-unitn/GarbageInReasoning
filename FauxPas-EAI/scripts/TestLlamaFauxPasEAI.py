from FauxPasEAIParser import FauxPasEAIParser
from FauxPasEvaluation import FauxPasEvaluation
from dotenv import load_dotenv
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

TEMPERATURE = None
NUMBER_OF_SAMPLES = 1
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

gt_answers = []
pred_answers = []

if __name__ == '__main__':
    model_name = MODEL_NAME
    temperature = TEMPERATURE
    number_of_samples = NUMBER_OF_SAMPLES
    out_str = ""
    print(model_name)
    out_str += model_name+"\n"
    print(temperature)
    out_str += "Temperature = " + str(temperature)+"\n"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    out_str += f"Using device: {device}\n"

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        do_sample=False,
        temperature=temperature,
        top_p=None
    )

    fp_parser = FauxPasEAIParser()
    for i in fp_parser.tests:
        print("**************************************")
        out_str += "**************************************\n"
        print(i)
        out_str += i + "\n"
        story, question1, answer1, question2, answer2, question3, answer3, question4, answer4 = fp_parser.tests[i]
        
        prompt1 = story + '\n' + question1 + '\nAnswer:'
        print(prompt1)
        out_str += prompt1 + "\n"
        for j in range(number_of_samples):
            response = pipe(prompt1)[0]['generated_text']
            s = response[len(prompt1):].strip()
            
            gt_answers.append(answer1)
            pred_answers.append(s)
            print("--------------------------------------")
            out_str += "--------------------------------------\n"
            print("ground truth: " + answer1 + " prediction: " + s)
            print(FauxPasEvaluation.compare_elements(answer1, s))
            out_str += "ground truth: " + answer1 + " prediction: " + s + "\n"
            out_str += str(FauxPasEvaluation.compare_elements(answer1, s)) + "\n"

        prompt2 = story + '\n' + question2 + '\nAnswer:'
        print(prompt2)
        out_str += prompt2 + "\n"
        for j in range(number_of_samples):
            response = pipe(prompt2)[0]['generated_text']
            s = response[len(prompt2):].strip().replace("\n", " ")
            
            gt_answers.append(answer2)
            pred_answers.append(s)
            print("--------------------------------------")
            out_str += "--------------------------------------\n"
            print("ground truth: " + answer2 + " prediction: " + s)
            print(FauxPasEvaluation.compare_elements_distance(answer2, s))
            out_str += "ground truth: " + answer2 + " prediction: " + s + "\n"
            out_str += str(FauxPasEvaluation.compare_elements_distance(answer2, s)) + "\n"

        prompt3 = story + '\n' + question3 + '\nAnswer:'
        print(prompt3)
        out_str += prompt3 + "\n"
        for j in range(number_of_samples):
            response = pipe(prompt3)[0]['generated_text']
            s = response[len(prompt3):].strip().replace("\n", " ")
            
            gt_answers.append(answer3)
            pred_answers.append(s)
            print("--------------------------------------")
            out_str += "--------------------------------------\n"
            print("ground truth: " + answer3 + " prediction: " + s)
            print(FauxPasEvaluation.compare_elements_distance(answer3, s))
            out_str += "ground truth: " + answer3 + " prediction: " + s + "\n"
            out_str += str(FauxPasEvaluation.compare_elements_distance(answer3, s)) + "\n"

        prompt4 = story + '\n' + question4 + '\nAnswer:'
        print(prompt4)
        out_str += prompt4 + "\n"
        for j in range(number_of_samples):
            response = pipe(prompt4)[0]['generated_text']
            s = response[len(prompt4):].strip()
            
            gt_answers.append(answer4)
            pred_answers.append(s)
            print("--------------------------------------")
            out_str += "--------------------------------------\n"
            print("ground truth: " + answer4 + " prediction: " + s)
            print(FauxPasEvaluation.compare_elements(answer4, s))
            out_str += "ground truth: " + answer4 + " prediction: " + s + "\n"
            out_str += str(FauxPasEvaluation.compare_elements(answer4, s)) + "\n"

    print("**************************************")
    out_str += "**************************************\n"
    print("Accuracy (question level):")
    out_str += "Accuracy (question level):"
    acc = FauxPasEvaluation.compare_lists_question_level(pred_answers, gt_answers)
    print(acc)
    out_str += str(acc)

    print("Accuracy (story level):")
    out_str += "Accuracy (story level):"
    acc = FauxPasEvaluation.compare_lists_story_level(pred_answers, gt_answers)
    print(acc)
    out_str += str(acc)

    with open(f"fp_{model_name.split('/')[-1]}_s-{number_of_samples}_t-{temperature}_acc-{acc}.txt", 'w') as f_out:
        f_out.write(out_str)
    with open(f"fp_{model_name.split('/')[-1]}_s-{number_of_samples}_predictions.txt", 'w') as f_out:
        f_out.write("\n".join(pred_answers)) 