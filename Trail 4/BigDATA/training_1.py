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

# Load your Excel files for training and validation
train_file_path = 'Data/Training DataSets.xlsx'

train_df = pd.read_excel(train_file_path)

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

# Apply text cleaning to 'Requirement Description' column for training data
train_df['Cleaned_Description'] = train_df['Requirement Description'].apply(clean_text)

# Use the same column names for target labels
y_train = train_df['Requirement Area']

# Tokenize and pad sequences for training data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['Cleaned_Description'])
X_train = tokenizer.texts_to_sequences(train_df['Cleaned_Description'])
X_train = pad_sequences(X_train, maxlen=100)

# Label encode 'Requirement Area' for training data
area_encoder = LabelEncoder()
y_train = area_encoder.fit_transform(y_train)

# Build an LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(area_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on training data
model.fit(X_train, y_train, epochs=150, batch_size=32)

# Save the trained model to an HDF5 file
model.save('TrainedLabelingModel.h5')

# Save the model as a Keras .keras file (optional)
model.save('TrainedLabelingModel.keras')
