import re
import ollama

class Guesser:
    def __init__(self, instruction, model_name, **kwargs):
        self.instruction = instruction
        self.model_name = model_name
        self.questions = []
        self.answers = [] 
        
    def add_question(self, question):
        self.questions.append(question)

    def qa2dict(self):
        qa_dict = {}
        for i, (question, answer) in enumerate(zip(self.questions, self.answers), 1):
            qa_dict[i] = {
                "question": question.text if hasattr(question, "text") else str(question),
                "answer": answer
            }
        return qa_dict

    def infer_answer(self):
        question = self.questions[-1]
        
        options_text = ""
        for i, opt in enumerate(question.options):
            options_text += f"[{i}] {opt.text}\n"
            
        user_content = f"Question: {question.text}\n{options_text}Answer: "
        
        try:
            response = ollama.chat(model=self.model_name, messages=[
                self.instruction, 
                {
                    'role': 'user',
                    'content': user_content,
                },
            ])
            response_text = response['message']['content']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            response_text = "0"

        match = re.search(r"\d", response_text)
        index = int(match.group()) if match else 0
        
        index = min(max(0, index), len(question.options) - 1)
        
        actual_id = question.options[index].id
        
        self.answers.append(actual_id)
        return actual_id
