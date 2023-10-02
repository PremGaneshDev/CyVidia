import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import load_model

# Load your Excel files for training and validation
train_file_path = 'Data/Training DataSets.xlsx'
validation_file_path = 'Data/Validation Dataset.xlsx'

train_df = pd.read_excel(train_file_path)
validation_df = pd.read_excel(validation_file_path)

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


# Apply text cleaning to 'Requirement Description' column  validation data
validation_df['Cleaned_Description'] = validation_df['Requirement Description'].apply(clean_text)

#Target labels for labeling model
y_train = train_df['Requirement Area']


# Tokenize and pad sequences for validation data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
X_val = tokenizer.texts_to_sequences(validation_df['Cleaned_Description'])
X_val = pad_sequences(X_val, maxlen=100)


# Load the trained model
model = load_model('TrainedLabelingModel.h5')

# Predict on validation data
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)


# Get prediction scores
prediction_scores = np.max(y_pred, axis=1)

# Create a threshold for labeling (e.g., 0.85)
labeling_threshold = 0.85

# Initialize lists to store predicted labels and suggested areas
predicted_labels = []
suggested_areas = []
area_encoder = LabelEncoder()
y_train = area_encoder.fit_transform(y_train)
# Iterate through predictions and apply labeling/suggestion logic
for pred, score in zip(y_pred_classes, prediction_scores):
    if score >= labeling_threshold:
        predicted_labels.append(area_encoder.inverse_transform([pred]))
        suggested_areas.append('')
    else:
        predicted_labels.append('Other')  # Label as 'Other' for scores < 85
        suggested_areas.append(area_encoder.inverse_transform([pred]))

# Add the predicted labels, prediction scores, and suggested areas to the validation DataFrame
validation_df['Predicted_Area'] = predicted_labels
validation_df['Prediction_Score'] = prediction_scores
validation_df['Suggested_Area'] = suggested_areas

# Save the validation results DataFrame to an Excel file
validation_results_file_path = 'Output/valdiation_1.2.xlsx'
validation_df.to_excel(validation_results_file_path, index=False)
print("Validation results with predictions saved to", validation_results_file_path)
