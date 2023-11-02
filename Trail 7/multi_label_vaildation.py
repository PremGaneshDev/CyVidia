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

# Load your Excel files for  validation
 
validation_file_path = '/Users/PremGanesh/Developer/Cyvidia/CyVidia/Input_Data/Validation Dataset.xlsx'
validation_df = pd.read_excel(validation_file_path)

# Load the saved tokenizer using pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Load the encoders from the training process
with open('area_encoder.pickle', 'rb') as handle:
    area_encoder = pickle.load(handle)
    
with open('bucket_encoder.pickle', 'rb') as handle:
    bucket_encoder = pickle.load(handle)

# Define a function for text cleaning
def clean_text(text):
    if isinstance(text, float) and np.isnan(text):
        return ""
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    clean_text = ' '.join(words)
    return clean_text

# Apply text cleaning to 'Requirement Description' column for validation data
validation_df['Cleaned_Description'] = validation_df['Requirement Description'].apply(clean_text)

# Tokenize and pad sequences for validation data
max_words = 1000
X_val = tokenizer.texts_to_sequences(validation_df['Cleaned_Description'])
X_val = pad_sequences(X_val, maxlen=100)

# Load the saved model
loaded_model = load_model('trained_model')



# Predict on validation data using the loaded model
y_pred = loaded_model.predict(X_val)

# Get prediction scores for 'Area' and 'Bucket'
prediction_scores_area = np.max(y_pred[0], axis=1)
prediction_scores_bucket = np.max(y_pred[1], axis=1)

# Add the prediction scores to the validation DataFrame
validation_df['Prediction_Score_Area'] = prediction_scores_area
validation_df['Prediction_Score_Bucket'] = prediction_scores_bucket

# Define a labeling threshold (e.g., 0.85)
labeling_threshold = 0.85

# Initialize lists to store predicted labels and suggested labels
predicted_labels_area = []
suggested_labels_area = []
predicted_labels_bucket = []
suggested_labels_bucket = []
predicted_2_labels_area = []

# Iterate through predictions and apply labeling/suggestion logic
for pred_area, score_area, pred_bucket, score_bucket in zip(
    np.argmax(y_pred[0], axis=1),
    prediction_scores_area,
    np.argmax(y_pred[1], axis=1),
    prediction_scores_bucket
):
    if score_area >= labeling_threshold:
        predicted_labels_area.append(area_encoder.inverse_transform([pred_area])[0])
        # predicted_2_labels_area.append(area_encoder.inverse_transform([pred_area])[1])
        suggested_labels_area.append('')

    else:
        predicted_labels_area.append('Other')  # Label as 'Other' for scores < 85
        suggested_labels_area.append(area_encoder.inverse_transform([pred_area])[0])

    if score_bucket >= labeling_threshold:
        predicted_labels_bucket.append(bucket_encoder.inverse_transform([pred_bucket])[0])
        suggested_labels_bucket.append('')
    else:
        predicted_labels_bucket.append('Other')  # Label as 'Other' for scores < 85
        suggested_labels_bucket.append(bucket_encoder.inverse_transform([pred_bucket])[0])

# Add the predicted labels and suggested labels to the validation DataFrame
validation_df['Predicted_Area'] = predicted_labels_area
validation_df['Suggested_Area'] = suggested_labels_area
validation_df['Predicted_Bucket'] = predicted_labels_bucket
validation_df['Suggested_Bucket'] = suggested_labels_bucket



# Save the validation results DataFrame to an Excel file
validation_results_file_path = '/Users/PremGanesh/Developer/Cyvidia/CyVidia/Output_Data/mlv.xlsx'
validation_df.to_excel(validation_results_file_path, index=False)
print("Validation results with predicted/suggested labels and scores saved to", validation_results_file_path)

