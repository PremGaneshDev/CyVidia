import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model   
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Load your Excel files for validation
validation_file_path = '/Users/PremGanesh/Developer/Cyvidia/CyVidia/Input_Data/Validation Dataset.xlsx'
validation_df = pd.read_excel(validation_file_path)
train_file_path = '/Users/PremGanesh/Developer/Cyvidia/CyVidia/Input_Data/Training DataSets.xlsx'
train_df = pd.read_excel(train_file_path)

# Load the saved tokenizer using pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#create clean_text function
def clean_text(text):
    if isinstance(text, float) and np.isnan(text):
        return ""
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    clean_text = ' '.join(words)
    return clean_text

# create the cleaned description column
validation_df['Cleaned_Description'] = validation_df['Requirement Description'].apply(clean_text)

# Target Labels for Training
y_train = train_df['Requirement Area']

# Label encode 'Requirement Area' for training data
area_encoder = LabelEncoder()
y_train = area_encoder.fit_transform(y_train)
# Tokenize and pad sequences for validation data
X_val = tokenizer.texts_to_sequences(validation_df['Cleaned_Description'])
X_val = pad_sequences(X_val, maxlen=100)

# Load the saved model
loaded_model = load_model('trained_model')  
# Predict on validation data using the loaded model
y_pred = loaded_model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get prediction scores
prediction_scores = np.max(y_pred, axis=1)

# Create a threshold for labeling (e.g., 0.85)
labeling_threshold = 0.85

# Initialize lists to store predicted labels and suggested areas
predicted_labels = []
suggested_areas = []

# Iterate through predictions and apply labeling/suggestion logic
for pred, score in zip(y_pred_classes, prediction_scores):
    if score >= labeling_threshold:
        predicted_labels.append(area_encoder.inverse_transform([pred])[0])
        suggested_areas.append('')
    else:
        predicted_labels.append('Other')  # Label as 'Other' for scores < 85
        suggested_areas.append(area_encoder.inverse_transform([pred])[0])

# Add the predicted labels, prediction scores, and suggested areas to the validation DataFrame
validation_df['Predicted_Area'] = predicted_labels
validation_df['Prediction_Score'] = prediction_scores
validation_df['Suggested_Area'] = suggested_areas

# Group the DataFrame by 'Predicted_Area'
grouped = validation_df.groupby('Predicted_Area')

# Create a dictionary to map group names to generalized questions
generalized_questions = {}

# Iterate through groups
for group_name, group_df in grouped:
    # Extract the cleaned descriptions for the group
    descriptions = group_df['Cleaned_Description'].tolist()

    # If there is only one description in the group, use it as the generalized question
    if len(descriptions) == 1:
        generalized_question = descriptions[0]
    else:
        # Use TF-IDF vectorization for text similarity
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

        # Calculate cosine similarity between descriptions
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Use Agglomerative Clustering to cluster similar questions
        clustering = AgglomerativeClustering(n_clusters=1, linkage='average', affinity='cosine')
        clustering.fit(cosine_similarities)

        # Extract the cluster labels
        cluster_labels = clustering.labels_

        # Find the most common cluster label (i.e., the cluster with the most questions)
        most_common_cluster_label = np.argmax(np.bincount(cluster_labels))

        # Find the questions belonging to the most common cluster
        most_common_cluster_indices = np.where(cluster_labels == most_common_cluster_label)[0]

        # Extract the questions in the most common cluster
        similar_questions = [descriptions[i] for i in most_common_cluster_indices]

        # Generate a generalized question by combining similar questions
        generalized_question = ' '.join(similar_questions)

        # Remove duplicate words from the generalized question 
        generalized_question = ' '.join(list(dict.fromkeys(generalized_question.split())))

        # frame the generalized question as a question (i.e., end with a question mark and capitalize the first letter) and meaning full qestion 
        generalized_question = generalized_question + '?' 

    # Add the generalized question to the dictionary
    generalized_questions[group_name] = generalized_question

# Add a new column 'Single_Phase_Question' to the original DataFrame with the generalized questions
validation_df['Single_Phase_Question'] = validation_df['Predicted_Area'].map(generalized_questions)

# sort the DataFrame by 'Predicted_Area' and 'Prediction_Score' a-z and descending
validation_df.sort_values(['Predicted_Area', 'Prediction_Score'], ascending=[True, False], inplace=True)


# Save the validation results DataFrame to an Excel file
validation_results_file_path = '/Users/PremGanesh/Developer/Cyvidia/CyVidia/Output_Data/grouping_with_single_phase.xlsx'
validation_df.to_excel(validation_results_file_path, index=False)
print("Validation results with predictions and single-phase questions saved to", validation_results_file_path)
