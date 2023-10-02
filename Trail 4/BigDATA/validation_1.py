import pandas as pd
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(filename='model_evaluation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load your validation data
validation_data_file = 'Data/Validation Dataset.xlsx'
validation_df = pd.read_excel(validation_data_file)

# Load the trained model
model = load_model('TrainedLabelingModel.h5')

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

# Load your training data and define y_train
train_file_path = 'Data/Training DataSets.xlsx'
train_df = pd.read_excel(train_file_path)
y_train = train_df['Requirement Area']  

# Tokenize and pad sequences for validation data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(validation_df['Cleaned_Description'])
X_val = tokenizer.texts_to_sequences(validation_df['Cleaned_Description'])
X_val = pad_sequences(X_val, maxlen=100)

# Label encode 'Requirement Area' for validation data
area_encoder = LabelEncoder()
area_encoder.fit(y_train)

# Make predictions on validation data
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
predicted_areas = area_encoder.inverse_transform(y_pred_classes)

# Add the predicted labels to the validation DataFrame
validation_df['Predicted_Area'] = predicted_areas

# Add the prediction level like score for each prediction to the validation DataFrame
validation_df['Prediction_Level'] = np.max(y_pred, axis=1)

# If prediction level is less than 0.8, set the predicted area to 'Other'
validation_df.loc[validation_df['Prediction_Level'] < 0.8, 'Predicted_Area'] = 'Other'

# Save the validation results DataFrame to an Excel file
validation_results_file_path = 'Output/ValidationResults_v1.xlsx'
validation_df.to_excel(validation_results_file_path, index=False)

