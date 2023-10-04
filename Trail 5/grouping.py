import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Load the context from an Excel file
file_path = '/Users/PremGanesh/Developer/Cyvidia/CyVidia/Output_Data/grouping_with_single_phase.xlsx'  # Replace 'your_excel_file.xlsx' with the actual file path
df = pd.read_excel(file_path)

# Extract the 'Context' column from the Excel file
context = ' '.join(df['Single_Phase_Question'].dropna())  # Assuming your context column is named 'Context'

# Tokenize the context into sentences (You might need to use a more advanced NLP library for better sentence splitting)
sentences = context.split('\n')

# Filter out empty or zero-length sentences
sentences = [sentence for sentence in sentences if sentence.strip()]

# Check if there are any non-empty sentences left
if not sentences:
    print("No non-empty sentences found in the context.")
else:
    # Use TF-IDF vectorization for text similarity
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    # Calculate cosine similarity between sentences
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Use Agglomerative Clustering to cluster similar sentences
    clustering = AgglomerativeClustering(n_clusters=1, linkage='average', affinity='cosine')
    clustering.fit(cosine_similarities)

    # Extract the cluster labels
    cluster_labels = clustering.labels_

    # Find the most common cluster label (i.e., the cluster with the most sentences)
    most_common_cluster_label = cluster_labels[0]

    # Find the sentences belonging to the most common cluster
    most_common_cluster_indices = [i for i, label in enumerate(cluster_labels) if label == most_common_cluster_label]

    # Extract the sentences in the most common cluster and join them to form the generalized question
    generalized_question = ' '.join([sentences[i] for i in most_common_cluster_indices])

    # Print the generated general question
    print(generalized_question)
