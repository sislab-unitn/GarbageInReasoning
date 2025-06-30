import openai

class LLMJudge:
    def __init__(self, api_key: str, model: str):
        """Initialize the LLM Judge with OpenAI API key.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model name
        """
        openai.api_key = api_key
        self.model = model

    def evaluate_answer(
        self,
        context: str,
        question: str,
        original_answer: str,
        predicted_answer: str,
        temperature: float = 0.0
    ) -> str:
        """Evaluate the predicted answer against the ground truth answer and the given question."""
        
        prompt = self._create_evaluation_prompt(
            context=context,
            question=question,
            original_answer=original_answer,
            predicted_answer=predicted_answer
        )

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in question answering systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=10
            )

            result = response.choices[0].message.content.strip()
            return result

        except Exception as e:
            raise Exception(f"Error during evaluation: {str(e)}")

    def _create_evaluation_prompt(
        self,
        context: str,
        question: str,
        original_answer: str,
        predicted_answer: str
    ) -> str:
        return f"""Your job is to evaluate a predicted answer by comparing it against the gold
        answer and the given question.
        You may refer to the provided context if needed.
        
        ## Grading Criteria:
        * 1: The predicted answer matches the gold answer or is a valid alternative (e.g.,different but correct ways of writing a name).
        * 0: The predicted answer is wrong or does not align with the gold answer.
        * In some ambiguous cases, where it is unclear whether the predicted answer is correct
        or not, please refer to the provided context and use it as the final source for making your
        judgment.
        
        ## Response Format:
        Please answer with ONLY '1' or '0'.
        
        ## Here is your task:
        Context: {context}
        
        Question: {question}
        
        Gold Answer: {original_answer}
        
        Predicted Answer: {predicted_answer}
        """