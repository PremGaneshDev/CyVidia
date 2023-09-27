import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score

# Load your Excel file
file_path = 'POC Based Requirements.xlsx'
df = pd.read_excel(file_path)

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

# Apply text cleaning to 'Requirement Description' column
df['Cleaned_Description'] = df['Requirement Description'].apply(clean_text)


# print(df['Requirement Description'])

# Split the data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(
    df['Cleaned_Description'], df['Requirement Area'], test_size=0.2, random_state=42
)

# Tokenize and pad sequences for training data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=100)

# Tokenize and pad sequences for validation data
X_val = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(X_val, maxlen=100)

# Label encode 'Requirement Area' for training and validation data
area_encoder = LabelEncoder()
y_train = area_encoder.fit_transform(y_train)
y_val = area_encoder.transform(y_val)

# Build an LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(area_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on training data
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Save the model to a file for future use 
model.save('model.h5')




# Predict on validation data
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# Create a DataFrame for the validation data with NEW_LABELS
validation_df = pd.DataFrame({'Requirement Description': X_val.tolist(), 'Requirement Area': y_val, 'NEW_LABELS': y_pred_classes})

# Convert all the binary values to human-readable format
validation_df['Requirement Area'] = area_encoder.inverse_transform(validation_df['Requirement Area'])
validation_df['NEW_LABELS'] = area_encoder.inverse_transform(validation_df['NEW_LABELS'])

# Convert the 'Requirement Description' column back to text with stopwords added
def convert_to_text(description):
    # Use the tokenizer's word index to reverse the sequence back to words
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    words = [reverse_word_index.get(word_idx, '') for word_idx in description]
    words_with_stopwords = [word for word in words if word not in ['', ' ']]
    return ' '.join(words_with_stopwords)

validation_df['Requirement Description'] = validation_df['Requirement Description'].apply(convert_to_text)


# save the validation data to set data to excel file add validation data to excel file


# Save the validation DataFrame to an Excel file
validation_file_path = 'Validation_Data_With_New_Labels.xlsx'
with pd.ExcelWriter(validation_file_path, engine='xlsxwriter') as writer:
    validation_df.to_excel(writer, sheet_name='Validation_Data', index=False)

print("Validation data with NEW_LABELS saved to", validation_file_path)


 