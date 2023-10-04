import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load your dataset from the Excel file
# data = pd.read_excel('Nist_for_single_phase.xlsx', engine='openpyxl')
data = pd.read_excel('Nist_for_single_phase.xlsx', engine='openpyxl')

# Initialize the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
   
# Tokenize your dataset
def tokenize_function(examples):
   return tokenizer(
      #string indices must be integers, not 'str' error is thrown if the column name is not a string type
        examples["Control Description"], return_length=True, padding="max_length", max_length=128, truncation=True
        )

# Tokenize the Control Descriptions

# ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.  





tokenized_descriptions = []
for description in data["Control Description"]:
    tokenized_description = tokenizer(description, return_length=True, padding="max_length", max_length=128, truncation=True)

    tokenized_descriptions.append(tokenized_description)

tokenized_datasets = data["Control Description"].map(tokenize_function)

# Define training arguments
training_args = TrainingArguments(
   per_device_train_batch_size=8,
   num_train_epochs=3,
   output_dir='./output',
)

# Initialize the trainer and train the model
trainer = Trainer(
   model=model,
   args=training_args,
   data_collator=tokenized_datasets.data_collator,
   train_dataset=tokenized_datasets.train_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model('./gpt2_question_generation_model')

# Generate questions based on control descriptions
control_descriptions = ["This is a sample control description.", "Another example control description."]
for desc in control_descriptions:
   input_text = "Generate a question based on the following control description:\n" + desc
   input_ids = tokenizer.encode(input_text, return_tensors="pt")

   # Generate the question
   generated_question = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
   decoded_question = tokenizer.decode(generated_question[0], skip_special_tokens=True)
   print(f"Control Description: {desc}")
   print(f"Generated Question: {decoded_question}\n")
