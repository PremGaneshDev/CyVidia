from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd
import nltk
import ssl
from gensim import corpora
import gensim
from nltk.corpus import stopwords
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Path to the Excel file
excel_file = "POC Based Requirements.xlsx"

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_file)

# Load the Requirements Bucket column into a list.
req_bucket = df['Requirements Bucket'].tolist()

# Load the Key Words column into a list.
req_keywords = df['Key Words'].tolist()

# Create a list of dictionaries in the desired format
data = [{"keywords": keywords.split(', '), "label": label} for keywords, label in zip(req_keywords, req_bucket)]

# Remove unnecessary columns
df = df.drop(['ReqNo', 'Requirement Area', 'Requirements Bucket', 'Key Words'], axis=1)

# Convert text to lowercase
df['Requirement Description'] = df['Requirement Description'].str.lower()

# Remove punctuation
df['Requirement Description'] = df['Requirement Description'].str.replace(r'[^\w\s]+', '')

# Download NLTK stopwords and 'punkt' if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Remove stopwords
stop = stopwords.words('english')
df['Requirement Description'] = df['Requirement Description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Remove numbers
df['Requirement Description'] = df['Requirement Description'].str.replace(r'\d+', '')

# Strip leading and trailing whitespaces
df['Requirement Description'] = df['Requirement Description'].str.strip()

# Tokenize the text
df['Requirement Description'] = df['Requirement Description'].apply(nltk.word_tokenize)

# Create a corpus from the tokenized text
text = df['Requirement Description'].tolist()
dictionary = corpora.Dictionary(text)
corpus = [dictionary.doc2bow(text) for text in text]

# Build the LDA model
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=len(data),
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

# Define the text classification pipeline
model = make_pipeline(CountVectorizer(analyzer=lambda x: x), MultinomialNB())

# Prepare the dataset
keywords = [item["keywords"] for item in data]
labels = [item["label"] for item in data]

# Train the text classification model
model.fit(keywords, labels)

# Predict labels for new requirements based on keywords
new_requirements = ["Incorporate/Ensure information security across all stages of application life cycle"]
predicted_labels = model.predict(new_requirements)
print("Predicted Label:", predicted_labels)