


import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load your Excel files for training and validation
train_file_path = '/Users/PremGanesh/Developer/Cyvidia/CyVidia/Input_Data/Training Dataset.xlsx'
train_df = pd.read_excel(train_file_path)

# Load your new data with additional labels
new_data_file_path = '/path/to/new_data.xlsx'
new_data_df = pd.read_excel(new_data_file_path)

# Combine the existing training data and new data
combined_train_df = pd.concat([train_df, new_data_df], ignore_index=True)

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

# Apply text cleaning to 'Requirement Description' column for combined training data
combined_train_df['Cleaned_Description'] = combined_train_df['Requirement Description'].apply(clean_text)

# Tokenize and pad sequences for combined training data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(combined_train_df['Cleaned_Description'])
X_train = tokenizer.texts_to_sequences(combined_train_df['Cleaned_Description'])
X_train = pad_sequences(X_train, maxlen=100)

# Label encode 'Requirement Area' for combined training data
area_encoder = LabelEncoder()
y_area_train = area_encoder.fit_transform(combined_train_df['Requirement Area (NIST)'])

# Label encode 'Requirement Bucket(NIST)' for combined training data
bucket_encoder = LabelEncoder()
y_bucket_train = bucket_encoder.fit_transform(combined_train_df['Requirement Bucket(NIST)'])

# Split data into training and validation sets for evaluation
# (You can adjust the validation split ratio as needed)
X_train, X_valid, y_area_train, y_area_valid, y_bucket_train, y_bucket_valid = train_test_split(
    X_train, y_area_train, y_bucket_train, test_size=0.2, random_state=42)

model_file_path = 'trained_model'  # Specify the path to the model file

if os.path.exists(model_file_path):
    # Load the existing model
    model = tf.keras.models.load_model(model_file_path)
else:
    # Create a new model
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=100))
    model.add(LSTM(128))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    output_area = Dense(len(area_encoder.classes_), activation='softmax', name='output_area')(model.layers[-1].output)
    output_bucket = Dense(len(bucket_encoder.classes_), activation='softmax', name='output_bucket')(model.layers[-1].output)
    model = Model(inputs=model.input, outputs=[output_area, output_bucket])
    model.compile(
        loss={'output_area': 'sparse_categorical_crossentropy', 'output_bucket': 'sparse_categorical_crossentropy'},
        optimizer='adam',
        metrics={'output_area': 'accuracy', 'output_bucket': 'accuracy'}
    )

# Train the model on the combined training data with both labels and store the training history
history = model.fit(X_train, {'output_area': y_area_train, 'output_bucket': y_bucket_train}, epochs=200, batch_size=32)

# Save the model to a file in the TensorFlow SavedModel format
model.save(model_file_path)

# Save the tokenizer using pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the encoders using pickle
with open('area_encoder.pickle', 'wb') as handle:
    pickle.dump(area_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bucket_encoder.pickle', 'wb') as handle:
    pickle.dump(bucket_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Evaluate the model on the validation set
y_area_pred, y_bucket_pred = model.predict(X_valid)

# visualize the confusion matrix for area prediction on validation data and calculate evaluation metrics for area prediction
area_pred = np.argmax(y_area_pred, axis=1)
area_valid = y_area_valid
cm_area = confusion_matrix(area_valid, area_pred)
print('Accuracy:', accuracy_score(area_valid, area_pred))
print('Precision:', precision_score(area_valid, area_pred, average='weighted'))
print('Recall:', recall_score(area_valid, area_pred, average='weighted'))
print('F1:', f1_score(area_valid, area_pred, average='weighted'))

# visualize the confusion matrix for bucket prediction on validation data and calculate evaluation metrics for bucket prediction
bucket_pred = np.argmax(y_bucket_pred, axis=1)
bucket_valid = y_bucket_valid
cm_bucket = confusion_matrix(bucket_valid, bucket_pred)
print('Accuracy:', accuracy_score(bucket_valid, bucket_pred))
print('Precision:', precision_score(bucket_valid, bucket_pred, average='weighted'))
print('Recall:', recall_score(bucket_valid, bucket_pred, average='weighted'))
print('F1:', f1_score(bucket_valid, bucket_pred, average='weighted'))

# Visualize the training progress
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['output_area_accuracy'])
plt.plot(history.history['output_bucket_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Area Accuracy', 'Bucket Accuracy'])
plt.subplot(1, 2, 2)
plt.plot(history.history['output_area_loss'])
plt.plot(history.history['output_bucket_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Area Loss', 'Bucket Loss'])
plt.axhline(y=0.5, color='r', linestyle='-')
plt.savefig('rbl_JLL_ciena.png')
