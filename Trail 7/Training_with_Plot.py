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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load your Excel files for training and validation
train_file_path = '/Users/PremGanesh/Developer/Cyvidia/CyVidia/Input_Data/Training Dataset.xlsx'
train_df = pd.read_excel(train_file_path)

#prinnt the shape of the training data
print(train_df.shape)

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

# Apply text cleaning to 'Requirement Description' column for training and validation data
train_df['Cleaned_Description'] = train_df['Requirement Description'].apply(clean_text)

# Tokenize and pad sequences for training data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['Cleaned_Description'])
X_train = tokenizer.texts_to_sequences(train_df['Cleaned_Description'])
X_train = pad_sequences(X_train, maxlen=100) # Pad the sequences to a maximum length of 100 words for training data 

# Label encode 'Requirement Area' for training data
area_encoder = LabelEncoder()
# Convert all the Requirement Area values to lower case
train_df['Requirement Area (NIST)'] = train_df['Requirement Area (NIST)'].str.lower()
y_area_train = area_encoder.fit_transform(train_df['Requirement Area (NIST)'])

# Label encode 'Requirement Bucket(NIST)' for training data
bucket_encoder = LabelEncoder()
# Convert all the Requirement Bucket values to lower case
train_df['Requirement Bucket(NIST)'] = train_df['Requirement Bucket(NIST)'].str.lower()
y_bucket_train = bucket_encoder.fit_transform(train_df['Requirement Bucket(NIST)'])

# Split data into training and validation sets for evaluation
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
    model.add(LSTM(128)) # Add an LSTM layer with 128 internal units
    model.add(Dense(64, activation='relu')) # Add a Dense layer with 64 units and ReLU activation function
    model.add(Dropout(0.5)) # Add a Dropout layer with 50% dropout rate
    output_area = Dense(len(area_encoder.classes_), activation='softmax', name='output_area')(model.layers[-1].output)
    output_bucket = Dense(len(bucket_encoder.classes_), activation='softmax', name='output_bucket')(model.layers[-1].output)
    model = Model(inputs=model.input, outputs=[output_area, output_bucket])
    model.compile(
        loss={'output_area': 'sparse_categorical_crossentropy', 'output_bucket': 'sparse_categorical_crossentropy'},
        optimizer='adam',
        metrics={'output_area': 'accuracy', 'output_bucket': 'accuracy'}
    )

# Train the model on training data with both labels and store the training history
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
 
import matplotlib.pyplot as plt

# Visualize label distribution for 'Area'
plt.figure(figsize=(10, 5))
plt.bar(area_encoder.classes_, train_df['Requirement Area (NIST)'].value_counts())
plt.xticks(rotation=60)
plt.xlabel('Area Labels')
plt.ylabel('Count')
# add title with labels count for each label
for i, v in enumerate(train_df['Requirement Area (NIST)'].value_counts()):
    plt.text(i - 0.2, v + 10, str(v))
#save the plot
plt.savefig('Label_Visualization.png')
plt.show()

# Visualize label distribution for 'Bucket'
plt.figure(figsize=(10, 5))
plt.bar(bucket_encoder.classes_, train_df['Requirement Bucket(NIST)'].value_counts())
plt.xticks(rotation=90)
plt.xlabel('Bucket Labels')
plt.ylabel('Count')
plt.title('Label Distribution for Bucket')
plt.show()
