import openai
from typing import Dict
import os
import json

class LLMRephraser:
    def __init__(self, api_key: str, model: str):
        """Initialize the LLM Rephraser with OpenAI API key.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model name
        """
        openai.api_key = api_key
        self.model = model

    def rephrase_item(
        self,
        item: Dict,
        temperature: float = 1.0
    ) -> str:
        """Query the model to generate 5 rephrasings of the given item.
        
        Args:
            item: The item to rephrase
            temperature: The temperature for the OpenAI API
        """
        required_keys = ["context", "question", "answerA", "answerB", "answerC"]
        missing_keys = [key for key in required_keys if key not in item]
        if missing_keys:
            raise ValueError(f"Missing required keys in item: {missing_keys}")
            
        context = item["context"]
        question = item["question"]
        answerA = item["answerA"]
        answerB = item["answerB"]
        answerC = item["answerC"]
        
        prompt = self._create_rephrasing_prompt(
            context=context,
            question=question,
            answerA=answerA,
            answerB=answerB,
            answerC=answerC
        )

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "You are an expert at rephrasing question-answering examples while maintaining their semantics and structure.\n\n" + prompt}
                ],
                temperature=temperature
            )

            result = response.choices[0].message.content.strip()
            return result

        except Exception as e:
            raise Exception(f"Error during rephrasing: {str(e)}")

    def _create_rephrasing_prompt(
        self,
        context: str,
        question: str,
        answerA: str,
        answerB: str,
        answerC: str
    ) -> str:
        return f"""Your job is to generate 5 different rephrasings of the given item.
        For each rephrasing, rephrase the context, question and possible answers while maintaining their semantics and structure.
        
        ## Response Format:
        Each rephrasing should be separated by "---REPHRASING---" and follow this exact format:
        
        REPHRASING 1:
        Context: [rephrased context]
        Question: [rephrased question]
        Answer A: [rephrased answer A]
        Answer B: [rephrased answer B]
        Answer C: [rephrased answer C]
        
        ---REPHRASING---
        
        REPHRASING 2:
        [same format as above]
        
        And so on for all 5 rephrasings.
        
        ## Original Item to Rephrase:
        Context: {context}
        
        Question: {question}
        
        Answer A: {answerA}
        Answer B: {answerB}
        Answer C: {answerC}
        """
        
def generate_rephrased_items(rephraser, data, checkpoint_interval=100, checkpoint_file="generated_rephrased_items/checkpoint.json"):
    """Generate rephrased items and save checkpoints periodically.
    
    Args:
        rephraser: LLMRephraser instance
        data: List of items to rephrase
        checkpoint_interval: Number of items to process before saving a checkpoint
        checkpoint_file: Path to the checkpoint file
    """
    results = []
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
            results = checkpoint_data.get("results", [])
            print(f"Loaded checkpoint with {len(results)} items")
    
    try:
        for i, item in enumerate(data, 1):
            if i <= len(results):
                print(f"Skipping item {i} (already processed)...")
                continue
                
            print(f"Generating rephrased items for item {i}...")
            rephrased_items = rephraser.rephrase_item(item)
            results.append(rephrased_items)
            
            if i % checkpoint_interval == 0:
                os.makedirs("generated_rephrased_items", exist_ok=True)
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump({"results": results}, f, indent=2, ensure_ascii=False)
                print(f"Saved checkpoint at item {i}/{len(data)}")
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Saving current progress...")
        os.makedirs("generated_rephrased_items", exist_ok=True)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        raise
    
    return results