from FauxPasEAIParser import FauxPasEAIParser
from FauxPasEvaluation import FauxPasEvaluation
import openai
from dotenv import load_dotenv
import os
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

TEMPERATURE = 0
NUMBER_OF_SAMPLES = 1
MODEL_NAME = "davinci-002"
#MODEL_NAME = "gpt-4-1106-preview", "gpt-3.5-turbo-1106", "davinci-002", "o1-preview"  #our queried models

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
    fp_parser = FauxPasEAIParser()
    for i in fp_parser.tests:
        print("**************************************")
        out_str += "**************************************\n"
        print(i)
        out_str += i + "\n"
        story, question1, answer1, question2, answer2, question3, answer3, question4, answer4 = fp_parser.tests[i]
        prompt1 = story + '\n' + question1 + '\nAnswer:'
        prompt2 = story + '\n' + question2 + '\nAnswer:'
        prompt3 = story + '\n' + question3 + '\nAnswer:'
        prompt4 = story + '\n' + question4 + '\nAnswer:'
        print(prompt1)
        out_str += prompt1 + "\n"
        conversation_history = []
        conversation_history_davinci = 'Story: ' + story + ' ' + question1 + ' Answer:'
        print(conversation_history_davinci)
        for j in range(number_of_samples):
            if MODEL_NAME=="gpt-3.5-turbo-1106" or MODEL_NAME=="gpt-4-1106-preview":
                time.sleep(3)
                try:
                    response = openai.chat.completions.create(
                        model=model_name,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt1}]
                    )
                except:
                    time.sleep(10)
                    response = openai.chat.completions.create(
                        model=model_name,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt1}]
                    )
                #print(response)
                s = response.choices[0].message.content
            elif MODEL_NAME=="o1-preview":
                response = openai.chat.completions.create(
                model="o1-preview",
                messages=[{"role": "user", "content": prompt1}],
                )
                s = response.choices[0].message.content
            else:
                response = openai.completions.create(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=1,
                    prompt=conversation_history_davinci
                )
                s = str(dict(response.choices[0])["text"]).strip()
            conversation_history.append({"role": "user", "content": prompt1})
            conversation_history.append({"role": "assistant", "content": s})
            conversation_history_davinci += s + ' ' 
            gt_answers.append(answer1)
            pred_answers.append(s)
            print("--------------------------------------")
            out_str += "--------------------------------------\n"
            print("ground truth: " + answer1 + " prediction: " + s)
            print(FauxPasEvaluation.compare_elements(answer1, s))
            out_str += "ground truth: " + answer1 + " prediction: " + s + "\n"
            out_str += str(FauxPasEvaluation.compare_elements(answer1, s)) + "\n"
        print(prompt2)
        out_str += prompt2 + "\n"
        conversation_history.append({"role": "user", "content": prompt2})
        conversation_history_davinci += question2 + ' Answer:'
        print('Before Q2:', conversation_history_davinci)
        for j in range(number_of_samples):
            if MODEL_NAME=="gpt-3.5-turbo-1106" or MODEL_NAME=="gpt-4-1106-preview":
                time.sleep(3)
                try:
                    response = openai.chat.completions.create(
                        model=model_name,
                        temperature=temperature,
                        messages=conversation_history
                    )
                except:
                    time.sleep(10)
                    response = openai.chat.completions.create(
                        model=model_name,
                        temperature=temperature,
                        messages=conversation_history
                    )
                    # print(response)
                s = response.choices[0].message.content
            elif MODEL_NAME=="o1-preview":
                response = openai.chat.completions.create(
                model="o1-preview",
                messages=conversation_history,
                )
                s = response.choices[0].message.content
            else:
                response = openai.completions.create(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=10,
                    prompt=conversation_history_davinci
                )
                s = str(dict(response.choices[0])["text"]).strip().replace("\n", " ") 
            gt_answers.append(answer2)
            pred_answers.append(s)
            print("--------------------------------------")
            out_str += "--------------------------------------\n"
            print("ground truth: " + answer2 + " prediction: " + s)
            print(FauxPasEvaluation.compare_elements_distance(answer2, s))
            out_str += "ground truth: " + answer2 + " prediction: " + s + "\n"
            out_str += str(FauxPasEvaluation.compare_elements_distance(answer2, s)) + "\n"
            conversation_history.append({"role": "assistant", "content": s})
            conversation_history_davinci += s + ' '
        print(prompt3)
        out_str += prompt3 + "\n"
        conversation_history.append({"role": "user", "content": prompt3})
        conversation_history_davinci += question3 + ' Answer:'
        print('Before Q3:', conversation_history_davinci)
        for j in range(number_of_samples):
            if MODEL_NAME=="gpt-3.5-turbo-1106" or MODEL_NAME=="gpt-4-1106-preview":
                time.sleep(3)
                try:
                    response = openai.chat.completions.create(
                        model=model_name,
                        temperature=temperature,
                        messages=conversation_history
                    )
                except:
                    time.sleep(10)
                    response = openai.chat.completions.create(
                        model=model_name,
                        temperature=temperature,
                        messages=conversation_history
                    )
                    # print(response)
                s = response.choices[0].message.content
            elif MODEL_NAME=="o1-preview":
                response = openai.chat.completions.create(
                model="o1-preview",
                messages=conversation_history,
                )
                s = response.choices[0].message.content
            else:
                response = openai.completions.create(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=10,
                    prompt=conversation_history_davinci
                )
                s = str(dict(response.choices[0])["text"]).strip().replace("\n", " ") 
            gt_answers.append(answer3)
            pred_answers.append(s)
            print("--------------------------------------")
            out_str += "--------------------------------------\n"
            print("ground truth: " + answer3 + " prediction: " + s)
            print(FauxPasEvaluation.compare_elements_distance(answer3, s))
            out_str += "ground truth: " + answer3 + " prediction: " + s + "\n"      #they had a typo here; fixed "answer1" to "answer3"
            out_str += str(FauxPasEvaluation.compare_elements_distance(answer3, s)) + "\n"
            conversation_history.append({"role": "assistant", "content": s})
            conversation_history_davinci += s + ' '
        print(prompt4)
        out_str += prompt4 + "\n"
        conversation_history.append({"role": "user", "content": prompt4})
        conversation_history_davinci += question4 + ' Answer:'
        print('Before Q4:', conversation_history_davinci)
        for j in range(number_of_samples):
            if MODEL_NAME=="gpt-3.5-turbo-1106" or MODEL_NAME=="gpt-4-1106-preview":
                time.sleep(3)
                try:
                    response = openai.chat.completions.create(
                        model=model_name,
                        temperature=temperature,
                        messages=conversation_history
                    )
                except:
                    time.sleep(10)
                    response = openai.chat.completions.create(
                        model=model_name,
                        temperature=temperature,
                        messages=conversation_history
                    )
                #print(response)
                s = response.choices[0].message.content
            elif MODEL_NAME=="o1-preview":
                response = openai.chat.completions.create(
                model="o1-preview",
                messages=conversation_history,
                )
                s = response.choices[0].message.content
            else:
                response = openai.completions.create(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=1,
                    prompt=conversation_history_davinci
                )
                s = str(dict(response.choices[0])["text"]).strip()
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
    acc= FauxPasEvaluation.compare_lists_question_level(pred_answers, gt_answers)
    print(acc)
    out_str += str(acc)

    print("Accuracy (story level):")
    out_str += "Accuracy (story level):"
    acc= FauxPasEvaluation.compare_lists_story_level(pred_answers, gt_answers)
    print(acc)
    out_str += str(acc)

    with open("C:/Users/lucyh/Documents/_PLIN/_UniTN/clever_hans/FauxPas-EAI/fp_conv_hist_"+model_name+"_s-"+str(number_of_samples)+"_t-"+str(temperature)+"_acc-"+str(acc)+"_story-only-once.txt", 'w') as f_out:
        f_out.write(out_str)
    with open("C:/Users/lucyh/Documents/_PLIN/_UniTN/clever_hans/FauxPas-EAI/fp_conv_hist_"+model_name+"_s-"+str(number_of_samples)+"predicitions_story-only-once.txt", 'w') as f_out:
        f_out.write("\n".join(pred_answers))
        
        
