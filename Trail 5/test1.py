import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model_name = "gpt2"  # You would use your fine-tuned model here
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define a function to generate questions with keywords and improved grammar
def generate_question(context, keyword):
    # Create the input text with context and keyword
    input_text = f" create a question for this context like easy to understand  :-{context}. {keyword} "
    
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the question
    output_ids = model.generate(input_ids, max_length=128, num_return_sequences=1)
    generated_question = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_question

# Example usage
context = "This control addresses the use of cryptography to protect the confidentiality and integrity of information. It includes specifying cryptographic key management requirements and the use of approved algorithms and cryptographic modules."
keyword = "Access control"
question = generate_question(context, keyword)
print(question)
