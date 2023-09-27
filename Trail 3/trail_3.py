# read the excel file POC Based Requirements.xlsx whuch contains the requirements for the POC project and it also have some keywords to understand the requirements based on that  ai models to understand the requirements and label them accordingly. And train a model that will label the requirements based on the keywords and the requirements description. 
# The excel file contains the following columns:
#    ReqNo
#	 Requirement Area	
#    Requirements Bucket
#	 Requirement Description
#	 Key Words	




import pandas as pd
 # pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. It is free software released under the three-clause BSD license. 
import nltk 
 # The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language. It was developed by Steven Bird and Edward Loper in the Department of Computer and Information Science at the University of Pennsylvania.
import ssl 
# The ssl module provides access to Transport Layer Security (often known as “Secure Sockets Layer”) encryption and peer authentication facilities for network sockets, both client-side and server-side.

# Path to the Excel file
excel_file = "POC Based Requirements.xlsx"

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_file)

# load the  Requirements Bucket column into a list.
req_bucket = df['Requirements Bucket']  
# print('req_bucket unshorted: ', req_bucket)
 

# the keywords in the list
req_keywords = df['Key Words']   
# print('req_keywords unshorted: ', req_keywords) 


# Create a list of dictionaries containing keywords and labels 
data = [{"keywords": keywords.split(', '), "label": label} for keywords, label in zip(req_keywords, req_bucket)] 

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




# Reading is done and now we will do the preprocessing of the data.


# Remove the columns
df = df.drop(['ReqNo', 'Requirement Area', 'Requirements Bucket', 'Key Words'], axis=1)

# Display the DataFrame
#print(df.head())
# add the labels_and_keywords to the dataframe df['requirement description'] column

print(df.head())   
# Now we will do the text preprocessing


# Convert the text to lowercase
df['Requirement Description'] = df['Requirement Description'].str.lower() 

# Remove punctuation
df['Requirement Description'] = df['Requirement Description'].str.replace(r'[^\w\s]+', '')



# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

from nltk.corpus import stopwords
# Download the stopwords if you haven't already uncomment nlkt.download('stopwords') and nltk.download('punkt') to download the stopwords and punkt
# nltk.download('stopwords')
# Download the 'punkt' resource
# nltk.download('punkt')


# Remove the stopwords
stop = stopwords.words('english')
df['Requirement Description'] = df['Requirement Description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Remove the numbers
df['Requirement Description'] = df['Requirement Description'].str.replace(r'\d+','')

# Remove leading and trailing whitespaces
df['Requirement Description'] = df['Requirement Description'].str.strip()

# Remove punctuation marks  like , . / \ etc
df['Requirement Description'] = df['Requirement Description'].str.replace(r'[^\w\s]+', '')



# Display the DataFrame
#print(df.head())

# Now we will do the tokenization of the data 


# Tokenize the text
df['Requirement Description'] = df['Requirement Description'].apply(nltk.word_tokenize)

# Display the DataFrame
# print(df.head())

# print the index and the tokenized data
# for index, row in df.iterrows():
#     print(index, row['Requirement Description'])






#For topic modeling is Latent Dirichlet Allocation (LDA). LDA is a generative probabilistic model that assumes each topic is a mixture over an underlying set of words, where each word in the document is attributable to the document's topics.

# Now we will do the topic modeling using LDA model
import gensim 
# Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community. Gensim is designed to handle large text collections using data streaming and incremental online algorithms, which differentiates it from most other scientific software packages that only target batch and in-memory processing.
from gensim import corpora 
# A corpus is a collection of texts. Such a collection is generally composed of a set of documents that may be anything from literary texts, medical journals, transcripts of speeches, to newspaper articles. We call the individual text a document and we call the entire collection a corpus.
 
# Create a corpus from a list of texts  
text = df['Requirement Description'].tolist() 
# print(text)


# Create a dictionary
dictionary = corpora.Dictionary(text)
# print(dictionary)
# print all the items in the dictionary  using for loop
# for key, value in dictionary.items():
#     print(key, ' : ', value)

# Term Document Frequency
corpus = [dictionary.doc2bow(text) for text in text]
# print(corpus)



# print(corpus[:1])
# # Human readable format of corpus (term-frequency)
# [[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]
 
# # Iterate through the corpus and print term-frequency representation for all requirements
# for i, doc in enumerate(corpus):
#     print(f"Term-frequency representation for Requirement {i + 1}:")
#     print([(dictionary[id], freq) for id, freq in doc])
#     print()


# Build the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,  # the corpus is the list of documents 
                                           id2word=dictionary,  # connect each word to its "spot" in the dictionary
                                           num_topics=len(labels_and_keywords),  #57 Adjust the number of topics as needed
                                           random_state=1,  # fix the seed for reproducibility
                                           update_every=10,  # how often the model should be updated with new documents
                                           chunksize=100,  # how many documents to use in each training chunk
                                           passes=10,  # how many times the entire corpus should be iterated 
                                           alpha='auto',  # the hyperparameter for LDA that governs sparsity of the document-topic (Theta) distributions
                                           per_word_topics=True,  # whether to return a representation of the model from which the word-topic matrix can be inferred
                                           )


# Print the Keyword in the  topics
# print('\n /\n printing keyword in the topics: ')
# print(lda_model.print_topics())
# doc_lda = lda_model[corpus]

# Compute Perplexity
# print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
# what is coherence score? The coherence score is a measure of how good a topic model is by calculating the degree of semantic similarity between high scoring words in a topic.
if __name__ == '__main__':
    
    from gensim.models import CoherenceModel
    coherence_model_lda = CoherenceModel(model=lda_model, texts=text, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)


 # Interpreting the Topics: Review the topics generated by LDA and assign human-readable labels to them based on your understanding of the keywords. These labels can help you understand the themes or categories that emerge from your requirements data.

# Visualize the topics-keywords
# Print the Keywords 
# print(lda_model.print_topics())

 
 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


 


# Prepare the dataset
keywords = [item["keywords"] for item in labels_and_keywords]
labels = [item["label"] for item in labels_and_keywords]

# Create a text classification pipeline
model = make_pipeline(CountVectorizer(analyzer=lambda x: x), MultinomialNB())
model.fit(keywords, labels)

# Topics generated by LDA model to human understandable labels using the text classification model
# Create a list of topics generated by LDA model to human understandable labels using the text classification model


topics_lda = lda_model.print_topics(num_topics=len(req_bucket)) 
# convert the topics_lda to the hunam readleble format 


# for topic_id, topic_keywords in topics_lda:
#     predicted_label = model.predict([topic_keywords])
#     print(f"Topic {topic_id}: {predicted_label}")
#     print("Keywords:", topic_keywords)
#     print()
    
# print('topics_lda: ', topics_lda)
# Assign labels to topics
# for topic_keywords in topics_lda:
#     predicted_label = model.predict([topic_keywords])
#     print("Keywords:", topic_keywords)
#     print("Predicted Label:", predicted_label)
#     print()

# extact newly generated labels from the topics_lda list create  a new excel file and add the new labels to the excel file and save it as a new excel file
# ...

# Create a new list of labels
new_labels = []
for topic_id, topic_keywords in topics_lda:
    predicted_label = model.predict([topic_keywords])[0]
    new_labels.append(predicted_label)
    print(f"Topic {topic_id}: {predicted_label}")
    print("Keywords:", topic_keywords)
    print()

# Ensure that new_labels has the same length as text
while len(new_labels) < len(text):
    new_labels.append(None)  # You can replace None with a default label if needed

# Create a new DataFrame
new_df = pd.DataFrame({'Requirement Description': text, 'Requirements Bucket': new_labels})
print(new_df.head())
print(new_df.tail())
print(new_df.shape)

# Save the DataFrame to an Excel file
new_df.to_excel('new_requirements.xlsx', index=False)
print('new_df: ', new_df)
