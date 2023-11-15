import spacy
from PyPDF2 import PdfReader
import pandas as pd
from textblob import TextBlob

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Increase the max_length limit (e.g., 2000000)
nlp.max_length = 2000000  # Set it to a value suitable for your input text

# Function to extract text from a PDF document 
def extract_text_from_pdf(pdf_file_path):
    pdf_text = ""
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

if __name__ == "__main__":
    pdf_file_path = "/Users/PremGanesh/Developer/AI/CyVidia/Input_Data/Claro.pdf" # Replace with the path to your PDF document
    
    pdf_text = extract_text_from_pdf(pdf_file_path)
    
    # Initialize a list to store sentence data
    sentence_data = []

    doc = nlp(pdf_text)
    for sentence in doc.sents:
        # Analyze or process each sentence as needed
        sentiment = TextBlob(sentence.text)
        sentiment_score = sentiment.sentiment.polarity
        if sentiment_score > 0:
            sentiment_label = "Positive"
        elif sentiment_score < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Append sentence data to the list
        sentence_data.append([sentence.text, sentiment_label, sentiment_score])

    # Create a DataFrame from the list
    df = pd.DataFrame(sentence_data, columns=['Sentence', 'Sentiment', 'Score'])
    # Remove rows with empty or null sentences
    df = df[df['Sentence'].notnull() & (df['Sentence'] != '')]

# Remove rows with short sentences (adjust the minimum length as needed)
    min_sentence_length = 10
    df = df[df['Sentence'].str.len() >= min_sentence_length]

# Remove rows with specific keywords (adjust the keywords as needed)
    invalid_keywords = ['⊠']
    df = df[~df['Sentence'].str.contains('|'.join(invalid_keywords), case=False)]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
# remove the white unwanted spaces from the sentence
    df['Sentence'] = df['Sentence'].str.strip()
    df['Sentence'] = df['Sentence'].str.replace('\n','')
    df['Sentence'] = df['Sentence'].str.replace('\r','')
    df['Sentence'] = df['Sentence'].str.replace('\t','')
    # Export to an Excel file
    df.to_excel('claro_requirement_data.xlsx', sheet_name='Sheet1', index=False)

     

