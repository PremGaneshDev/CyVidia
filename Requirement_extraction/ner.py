import spacy
from spacy.training.example import Example
import pandas as pd

# Load your spaCy model
nlp = spacy.blank("en")

# Load your Excel file into a DataFrame
excel_file_path = "/Users/PremGanesh/Developer/AI/CyVidia/Input_Data/nist for extraction.xlsx"
df = pd.read_excel(excel_file_path)

# Define your entity labels
labels = ['CONTROL_IDENTIFIER', 'CONTROL_MAIN_AREA', 'CONTROL_BUCKET', 'CONTROL_SUB_BUCKETS', 'DISCUSSION', 'RELATED_CONTROLS']

# Add NER to the pipeline
ner = nlp.add_pipe("ner", config={"labels": labels}, last=True)

# Prepare training data
training_data = []

# Iterate over rows in the DataFrame
for index, row in df.iterrows():
    text = str(row['Discussion'])  # Use the column containing text for training, adjust as needed

    # Extract entity annotations from the row
    entities = []
    for label in labels:
        start = row[label + '_start']  # Adjust these column names based on your Excel file
        end = row[label + '_end']
        entities.append((start, end, label))

    example = Example.from_dict(nlp.make_doc(text), {"entities": entities})
    training_data.append(example)

# Train the NER model
nlp.begin_training()
for epoch in range(10):  # Adjust the number of epochs as needed
    for example in training_data:
        nlp.update([example], drop=0.5)  # Adjust dropout as needed

# Save the trained model
nlp.to_disk("path/to/your/trained_model")
