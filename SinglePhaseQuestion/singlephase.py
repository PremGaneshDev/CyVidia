import pandas as pd
import openai

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key = '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'

# Read the data from the Excel file
data = pd.read_excel('/Users/PremGanesh/Developer/AI/CyVidia/Input_Data/RBI_training data.xlsx', sheet_name='Dashboard sheet')

#load the first 30 only 
data = data.head(30)
# Initialize the OpenAI API client
openai.api_key = api_key

# Create a list to store the generated Yes/No questions
yes_no_questions = []

# Generate Yes/No questions for each requirement description
for index, row in data.iterrows():
    requirement_description = row['Requirement Description']
    
    # Generate a Yes/No question for the description
    question = f"As the customer , should the client follow these rules: {requirement_description} ?"
    # use the openai api to generate the question
    response = openai.Completion.create(
        engine='davinci',
        prompt=question,
        temperature=0.3,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['\n']
    )
    # Get the question from the response
    question = response['choices'][0]['text']
    
    # Append the question to the list
    yes_no_questions.append({'Question': question})

# Create a DataFrame from the list of questions
questions_df = pd.DataFrame(yes_no_questions)

# Add the questions to the original DataFrame
data['Yes/No Question'] = questions_df['Question']


print(data['Yes/No Question'])
# Save the questions to the original Excel file
data.to_excel('Traning dataset_with_questions.xlsx', index=False, sheet_name='Dashboard sheet')
