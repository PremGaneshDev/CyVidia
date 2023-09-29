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

# Apply text cleaning to 'Requirement Description' column for training and validation data
train_df['Cleaned_Description'] = train_df['Requirement Description'].apply(clean_text)
validation_df['Cleaned_Description'] = validation_df['Requirement Description'].apply(clean_text)

# Use the same column names for target labels
y_train = train_df['Requirement Area']
y_val = validation_df['Requirement Description']

# Tokenize and pad sequences for training data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['Cleaned_Description'])
X_train = tokenizer.texts_to_sequences(train_df['Cleaned_Description'])
X_train = pad_sequences(X_train, maxlen=100)

# Tokenize and pad sequences for validation data
X_val = tokenizer.texts_to_sequences(validation_df['Cleaned_Description'])
X_val = pad_sequences(X_val, maxlen=100)

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


# save the trained model in h5 file 
# model.save('TrainedLabelingModel.h5')

model.save('TrainedLabelingModel.keras')











# print(f"Validation results with predictions saved to {validation_results_file_path}")
# print(f"Number of predictions: {validation_df.shape[0]}")
# print(f"Number of unique predictions: {validation_df['Predicted_Area'].nunique()}")
# print(f"Unique predictions: {validation_df['Predicted_Area'].unique()}")
# print(f"Number of unique requirement areas: {validation_df['Requirement Area'].nunique()}")
# print(f"Unique requirement areas: {validation_df['Requirement Area'].unique()}")
# print(f"Number of correct predictions: {validation_df['Requirement Area'].eq(validation_df['Predicted_Area']).sum()}")
# print(f"Number of incorrect predictions: {validation_df['Requirement Area'].ne(validation_df['Predicted_Area']).sum()}")
# print(f"Accuracy: {validation_df['Requirement Area'].eq(validation_df['Predicted_Area']).mean() * 100}%")



# merged_df = pd.merge(train_df, validation_df, on="Requirement Description", how="inner")

# # Check if Predicted_Area matches Requirement Area and create a "Correct" column
# merged_df['Correct'] = merged_df.apply(lambda row: 'Valid' if row['Predicted_Area'] == row['Requirement Area'] else 'Invalid', axis=1)


# # Print the number of correct and incorrect predictions
# print(f"Number of correct predictions: {merged_df['Correct'].eq('Valid').sum()}")
# print(f"Number of incorrect predictions: {merged_df['Correct'].eq('Invalid').sum()}")
# print(f"Total number of predictions: {merged_df.shape[0]}")

# # Print the accuracy of the model on the validation data set
# print(model.accuracy(X_val, y_val))


# print(f"Accuracy: {merged_df['Correct'].eq('Valid').mean() * 100}%")

# # Create a new Excel file with the "Correct" column
# output_excel = "Validation_Results_With_Correctness.xlsx"  
# merged_df.to_excel(output_excel, index=False, engine='openpyxl')

# print(f"Validation results with correctness saved to {output_excel}")
