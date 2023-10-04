import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load your data from an Excel file
# Replace 'your_data.xlsx' with the actual path to your Excel file
data = pd.read_excel('/Users/PremGanesh/Developer/Cyvidia/CyVidia/Output_Data/grouping_with_single_phase.xlsx')

# Initialize the T5 model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Define a function to generate questions using the T5 model
def generate_question(single_phase_question, predicted_area):
    # Create the input text in the format expected by T5
    input_text = f"Generate a question  {predicted_area}: {single_phase_question}"
    
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the question using the T5 model conditioned on the input text  qestion 
    output_ids = model.generate(input_ids, max_length=64, num_return_sequences=1) # type: ignore
    generated_question = tokenizer.decode(output_ids[0], skip_special_tokens=True)
   
    
    return generated_question

# Create an empty list to store generated questions
questions = []

# Iterate through the rows of the DataFrame
for index, row in data.iterrows():
    single_phase_question = row['Single_Phase_Question']
    predicted_area = row['Predicted_Area']
    
    # Generate the question for this row
    question = generate_question(single_phase_question, predicted_area)
    
    # Append the question to the list
    questions.append(question)

# Add the generated questions to the DataFrame
data['Generated_Question'] = questions

# Save the DataFrame with the generated questions to a new Excel file
# Replace 'output_data.xlsx' with the desired output file name
data.to_excel('/Users/PremGanesh/Developer/Cyvidia/CyVidia/Output_Data/output_data.xlsx', index=False)












