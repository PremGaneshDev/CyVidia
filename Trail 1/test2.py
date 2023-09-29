import pandas as pd
import nltk
import ssl
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load the Excel file into a DataFrame
excel_file = "POC Based Requirements.xlsx"
df = pd.read_excel(excel_file)

# Preprocessing of Requirement Description
df['Requirement Description'] = df['Requirement Description'].str.lower()
df['Requirement Description'] = df['Requirement Description'].str.replace(r'[^\w\s]+', '')
nltk.download('stopwords')
nltk.download('punkt')
stop = nltk.corpus.stopwords.words('english')
df['Requirement Description'] = df['Requirement Description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['Requirement Description'] = df['Requirement Description'].str.replace(r'\d+', '')
df['Requirement Description'] = df['Requirement Description'].str.strip()
df['Requirement Description'] = df['Requirement Description'].apply(nltk.word_tokenize)

# Handling missing values by filling NaN with empty string
df['Key Words'].fillna('', inplace=True)

# Load the Keywords and Labels into lists
keywords = df['Key Words'].str.split(', ')
labels = df['Requirements Bucket']

# Create a text classification pipeline
model = make_pipeline(CountVectorizer(analyzer=lambda x: x), MultinomialNB())
model.fit(keywords, labels)

# Topic Modeling using LDA
text = df['Requirement Description'].tolist()
dictionary = corpora.Dictionary(text)
corpus = [dictionary.doc2bow(text) for text in text]

# Build the LDA model
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=6, random_state=100, passes=10, alpha='auto', per_word_topics=True)

# Assign topics to requirements
topics = lda_model.get_document_topics(corpus)
for i, topic in enumerate(topics):
    topic = sorted(topic, key=lambda x: x[1], reverse=True)
    top_topic = topic[0][0]
    top_topic_keywords = lda_model.print_topic(top_topic)
    predicted_label = model.predict([top_topic_keywords])[0]
    print(f"Requirement {i+1}: Predicted Label - {predicted_label}, Topic - {top_topic}")
