import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Parameters
MAX_NUM_WORDS = 10000  # Max number of words in the tokenizer
MAX_SEQ_LENGTH = 100  # Reduced max sequence length for padding
EMBEDDING_DIM = 128  # Dimension of word embeddings

# Load data
data = pd.read_excel('POC Based Requirements.xlsx')

# Use 'Requirement Description' as text data and 'Requirements Bucket' as labels
text_data = data['Requirement Description']
labels = data['Requirements Bucket']

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(text_data)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(text_data)
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')

# Convert labels to one-hot vectors
labels_index = {category: idx for idx, category in enumerate(set(labels))}
label_count = len(labels_index)

labels = to_categorical(np.array(labels.map(labels_index)))

# Split your data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the LSTM model with increased complexity and regularization
model = Sequential([
    Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH),
    Bidirectional(LSTM(128, return_sequences=True)),  # Increased units
    Bidirectional(LSTM(64)),  # Increased units
    Dense(64, activation='relu'),
    Dropout(0.5),  # Adjust dropout rate
    Dense(label_count, activation='softmax')
])

# Experiment with learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model with a larger number of epochs
model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val), batch_size=32)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
