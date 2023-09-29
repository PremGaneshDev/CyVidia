# Install necessary packages
#pip install openai tenacity pandas

import os
import openai
import random
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import pandas as pd

# Set your OpenAI API key
openai.api_key = "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"

# Define the system message for fine-tuning
system_message = "the requirements for the POC project and it also have some keywords to understand the requirements based on the user ai models to understand the requirements and label them accordingly. and train a model that will label the requirements based on the keywords and the requirements description. "

# Remove duplicates
df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples.')

# Initialize list to store training examples
training_examples = []

# Create training examples in the format required for GPT-3.5 fine-tuning
for index, row in df.iterrows():
    training_example = {
        "messages": [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": row['response']}
        ]
    }
    training_examples.append(training_example)

# Save training examples to a .jsonl file
with open('training_examples.jsonl', 'w') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\n')

# Create a file using OpenAI File API
file_id = openai.File.create(
    file=open("training_examples.jsonl", "rb"),
    purpose='fine-tune'
).id

# Create a fine-tuning job
job = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")

job_id = job.id
# List events for the fine-tuning job
events = openai.FineTuningJob.list_events(id=job_id, limit=10)
print(events)

# Retrieve the fine-tuned model name
model_name_pre_object = openai.FineTuningJob.retrieve(job_id)
model_name = model_name_pre_object.fine_tuned_model
print(model_name)

# Use the fine-tuned model to generate a response
response = openai.ChatCompletion.create(
    model='ft:gpt-3.5-turbo-0613:krify-co::7yyga2va',
    messages=[
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": 'Incorporate/Ensure information security across all stages of application life cycle',
        }
    ],
)

# Get the generated response content
generated_response = response.choices[0].message['content']
print(generated_response)
