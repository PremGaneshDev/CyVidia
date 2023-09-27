import gensim
from gensim import corpora

# Define your documents
documents = [
    '''REs shall follow a ‘secure by design’ approach in the development of digital payment products and services. REs shall ensure that digital payment applications are inherently more secure by embedding security within their development lifecycle.''',

     "Besides business functionalities, security requirements relating to system access control, authentication, transaction authorization, data integrity, system activity logging, audit trail, session management, security event tracking and exception handling are required to be clearly specified at the initial and ongoing stages of system development/acquisition/implementation.",
    
       "In respect of critical business applications, banks may consider conducting source code audits by professionally competent personnel/service providers or have assurance from application providers/OEMs that the application is free from embedded malicious / fraudulent code",
       
    "Incorporate/Ensure information security across all stages of application life cycle",
 
    "Secure coding practices may also be implemented for internally /collaboratively developed applications.",
   
]

# Tokenize the documents (split them into words)
tokenized_docs = [doc.split() for doc in documents]

# Create a dictionary from the tokenized documents
dictionary = corpora.Dictionary(tokenized_docs)

# Create a corpus (bag of words) from the tokenized documents
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Define your keywords and labels for training
keywords_and_labels = [
    (["Ensure", "stage", "application", "life", "cycle"], "Secure SDLC Lifecycle"),
    (["Professionally", "service", "providers", "code", "audits", "assurance"], "Third Party Code Audits"),
    (["secure", "coding", "practices", "internally", "applications"], "Secure Coding Practices"),
    (["system", "access", "control", "authentication", "transaction", "authorization", "data", "integrity", "system", "activity", "logging", "audit", "trail", "session", "management", "security", "event", "tracking", "exception", "handling"], "Security Requirements"),
]

# Flatten the keywords for training
all_keywords = [keyword for keywords, _ in keywords_and_labels for keyword in keywords]

# Train the LDA model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                    id2word=dictionary,
                                    num_topics=len(keywords_and_labels),  # Number of topics based on your labels
                                    random_state=100,
                                    update_every=1,
                                    chunksize=10,
                                    passes=10,
                                    alpha='auto',
                                    per_word_topics=True)

# Define topic labels based on your interpretation of the topics
topic_labels = {
    i: label for i, (_, label) in enumerate(keywords_and_labels)
}

# Assign each keyword to a topic and print the results
for keyword, label in keywords_and_labels:
    keyword_bow = dictionary.doc2bow(keyword)
    topic = lda_model[keyword_bow][0][0]  # Get the most dominant topic for the keyword
    topic_id = topic[0]  # Extract the topic ID from the tuple
    print(f"Keyword {keyword} belongs to topic: {topic_labels[topic_id]}")
