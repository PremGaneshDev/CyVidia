# read the excel file POC Based Requirements.xlsx whuch contains the requirements for the POC project and it also have some keywords to understand the requirements based on the user ai models to understand the requirements and label them accordingly. and train a model that will label the requirements based on the keywords and the requirements description. 
# The excel file contains the following columns:
#    ReqNo
#	 Requirement Area	
#    Requirements Bucket
#	 Requirement Description
#	 Key Words	

import pandas as pd

# Path to the Excel file
excel_file = "POC Based Requirements.xlsx"

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_file)

# load the  Requirements Bucket column into a list.
req_bucket = df['Requirements Bucket']   # type: ignore
# print('req_bucket unshorted: ', req_bucket)
 

# same as req_bucket but sorted short the keywords in the list
req_keywords = df['Key Words']   # type: ignore
# print('req_keywords unshorted: ', req_keywords) 

# Create a list of dictionaries in the desired format
data = [{"keywords": keywords.split(', '), "label": label} for keywords, label in zip(req_keywords, req_bucket)] # type: ignore

# Create a dictionary to store merged keywords
merged_keywords_dict = {}

# Iterate through the 'data' list and merge keywords for topics with the same label
for item in data:
    label = item['label']
    keywords = item['keywords']
    
    # If the label is already in the dictionary, append the keywords
    if label in merged_keywords_dict:
        merged_keywords_dict[label].extend(keywords)
    else:
        merged_keywords_dict[label] = keywords

# Create a new list of dictionaries with merged keywords
labels_and_keywords = [{"keywords": keywords, "label": label} for label, keywords in merged_keywords_dict.items()]
# print('merged_data list: ', labels_and_keywords)

# reading is done and now we will do the preprocessing of the data
# Remove the columns
df = df.drop(['ReqNo', 'Requirement Area', 'Requirements Bucket', 'Key Words'], axis=1)

# Display the DataFrame
#print(df.head())

# Now we will do the text preprocessing
# Convert the text to lowercase
df['Requirement Description'] = df['Requirement Description'].str.lower() 

# Remove punctuation
df['Requirement Description'] = df['Requirement Description'].str.replace(r'[^\w\s]+', '')



import nltk
import ssl
# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context
# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
# Download the stopwords if you haven't already
nltk.download('stopwords')
# Now you can use the stopwords
stop = stopwords.words('english')
# Remove the stopwords
stop = stopwords.words('english')
df['Requirement Description'] = df['Requirement Description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Remove the numbers
df['Requirement Description'] = df['Requirement Description'].str.replace(r'\d+','')

# Remove leading and trailing whitespaces

df['Requirement Description'] = df['Requirement Description'].str.strip()

# Display the DataFrame
#print(df.head())

# Now we will do the tokenization of the data 
# Tokenize the text

# Download the 'punkt' resource
nltk.download('punkt')

df['Requirement Description'] = df['Requirement Description'].apply(nltk.word_tokenize)

# Display the DataFrame
#print(df.head())

# for topic modeling is Latent Dirichlet Allocation (LDA). LDA is a generative probabilistic model that assumes each topic is a mixture over an underlying set of words, where each word in the document is attributable to the document's topics.

# Now we will do the topic modeling using LDA

import gensim
from gensim import corpora
 
# Create a corpus from a list of texts  
text = df['Requirement Description'].tolist() # type: ignore
#print(text)
# Create a dictionary
dictionary = corpora.Dictionary(text)
#print(dictionary)
# Term Document Frequency
corpus = [dictionary.doc2bow(text) for text in text]
#print(corpus)
# View
#print(corpus[:1])
# Human readable format of corpus (term-frequency)
[[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]

# Build the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,# the corpus is the list of documents
                                           id2word=dictionary, # connect each word to its "spot" in the dictionary
                                           num_topics= len(labels_and_keywords),  # Adjust the number of topics as needed
                                           random_state=100, # fix the seed for reproducibility
                                           update_every=1, # how often the model should be updated with new documents
                                           chunksize=100, # how many documents to use in each training chunk
                                           passes=10, # how many times the entire corpus should be iterated 

                                           alpha='auto', # the hyperparameter for LDA that governs sparsity of the document-topic (Theta) distributions
                                           per_word_topics=True, # whether to return a representation of the model from whicch the word-topic matrix can be inferred
                                             )

# Print the Keyword in the  topics
# print('\n /\n printing keyword in the topics: ')
# print(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
# what is coherence score? The coherence score is a measure of how good a topic model is by calculating the degree of semantic similarity between high scoring words in a topic.
if __name__ == '__main__':
    
    from gensim.models import CoherenceModel
    coherence_model_lda = CoherenceModel(model=lda_model, texts=text, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


 # Interpreting the Topics: Review the topics generated by LDA and assign human-readable labels to them based on your understanding of the keywords. These labels can help you understand the themes or categories that emerge from your requirements data.

# Visualize the topics-keywords
# Print the Keywords in the 5 topics
# print(lda_model.print_topics())
 

 

# # Print the Keywords in the 5 topics
# topics = lda_model.print_topics(num_topics=5, num_words=10)  # Adjust the number of words as needed
# for topic_id, topic_keywords in topics:
#     print(f"Topic {topic_id}:")
#     print(topic_keywords)
#     print()  # Add a line break between topics


# Define human-readable topic labels based on the keywords assigned to each topic by the LDA model. You can use these labels to interpret the topics. 
# i have topic labels in req_bucket list and i will assign them to the topics
# Define the topic labels
# use the req_bucket list to assign the labels to the topics

# topic_labels = req_bucket
# print(topic_labels)

# # Print the Topics with Labels
# topics = lda_model.print_topics(num_topics=5, num_words=10)
# for topic_id, topic_keywords in topics:
#     print(f"Topic {topic_id}: {topic_labels[topic_id]}")
#     print("Keywords:", topic_keywords)
#     print()


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


 


# Prepare the dataset
keywords = [item["keywords"] for item in data]
labels = [item["label"] for item in data]

# Create a text classification pipeline
model = make_pipeline(CountVectorizer(analyzer=lambda x: x), MultinomialNB())
model.fit(keywords, labels)

# Topics generated by LDA model to human understandable labels using the text classification model
# Create a list of topics generated by LDA model to human understandable labels using the text classification model

topics_lda = lda_model.print_topics(num_topics=len(req_bucket), num_words=10) # Adjust the number of words as needed 
# convert the topics_lda to the hunam readleble format 


for topic_id, topic_keywords in topics_lda:
    predicted_label = model.predict([topic_keywords])[0]
    # print(f"Topic {topic_id}: {predicted_label}")
    # print("Keywords:", topic_keywords)
    # print()
    

# topics_lda = [
#     ["data", "access", "information", "controls", "systems"],
#     ["data", "vulnerabilities", "application", "critical", "business", "security", "sensitive"],
#     # Add more topics...
# ]

# Assign labels to topics
for topic_keywords in topics_lda:
    predicted_label = model.predict([topic_keywords])[0]
    print("Keywords:", topic_keywords)
    print("Predicted Label:", predicted_label)
    print()

 


 