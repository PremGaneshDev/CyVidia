# Cell 6
# Import necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
 

# Load your Excel file with sentences
excel_file_path = '/Users/PremGanesh/Developer/AI/CyVidia/Requirement_extraction/claro_requirement_data.xlsx'
df = pd.read_excel(excel_file_path)

# Define a function for text cleaning
def clean_text(text):
    if isinstance(text, float) and np.isnan(text):  # check if the text is NaN
        return ""  # return an empty string
    words = word_tokenize(text)  # tokenize the text into words using NLTK library
    words = [word.lower() for word in words if word.isalnum()]  # convert all words to lowercase and remove punctuations
    stop_words = set(stopwords.words('english'))  # get the stop words from NLTK library
    words = [word for word in words if word not in stop_words]  # remove stop words from the text
    clean_text = ' '.join(words)  # join all words into a sentence
    return clean_text  # return the cleaned text

# Apply text cleaning to the 'Sentence' column
df['Cleaned_Sentence'] = df['Sentence'].apply(clean_text)

# Tokenize and pad sequences for the sentences
tokenized_sentences = tokenizers.texts_to_sequences(df['Cleaned_Sentence'])
padded_sentences = pad_sequences(tokenized_sentences, maxlen=100)

# Make predictions
sentence_predictions = model.predict(padded_sentences)

# Assuming your model outputs probabilities, you can threshold them to get binary predictions
threshold = 0.5
binary_sentence_predictions = [(area > threshold).astype(int) for area in sentence_predictions]

# Decode the predictions using the label encoders
sentence_labels = [area_encoder.classes_[prediction[0]] for prediction in binary_sentence_predictions]

# Add the predicted labels to the DataFrame
df['Related_to_Cybersecurity'] = sentence_labels

# Print the DataFrame with predictions
print(df[['Sentence', 'Related_to_Cybersecurity']])
