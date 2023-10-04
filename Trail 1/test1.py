import pandas as pd
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import nltk
import ssl

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Download stopwords and 'punkt' resource if you haven't already
nltk.download('stopwords')
nltk.download('punkt')

# Define and use stopwords
stop = stopwords.words('english')

def main():
    # Load the Excel file into a DataFrame
    excel_file = "/Users/PremGanesh/Developer/Cyvidia/CyVidia/Input_Data/Training DataSets.xlsx"
    df = pd.read_excel(excel_file)

    # Data Preprocessing
    df['Requirement Description'] = df['Requirement Description'].str.lower()
    df['Requirement Description'] = df['Requirement Description'].str.replace(r'[^\w\s]+', '')
    df['Requirement Description'] = df['Requirement Description'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop])
    )
    df['Requirement Description'] = df['Requirement Description'].str.replace(r'\d+', '')
    df['Requirement Description'] = df['Requirement Description'].str.strip()
    df['Requirement Description'] = df['Requirement Description'].apply(nltk.word_tokenize)

    # Topic Modeling (LDA)  Latent Dirichlet Allocation 
    text = df['Requirement Description'].tolist()
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(text) for text in text]

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=5,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True,
        workers=1  # Specify the number of CPU cores to use
    )

    # Print the Keywords in the 5 topics
    print(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Compute Perplexity (a lower value is better)
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))

    # Compute Coherence Score (a higher value is better)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=text, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

if __name__ == "__main__":
    main()
