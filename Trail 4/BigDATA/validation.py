import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
 



# Load your validation data
validation_data_file = 'Data/Validation Dataset.xlsx'
validation_df = pd.read_excel(validation_data_file)

# Load the trained model
model = load_model('TrainedLabelingModel.keras')

# Load your training data and define y_train
train_file_path = 'Data/Training DataSets.xlsx'
train_df = pd.read_excel(train_file_path)
y_train = train_df['Requirement Area']  # Adjust this line to match your actual training data structure

# Tokenize and pad sequences for validation data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(validation_df['Requirement Description'])
X_val = tokenizer.texts_to_sequences(validation_df['Requirement Description'])
X_val = pad_sequences(X_val, maxlen=100)

# Label encode 'Requirement Area' for validation data
area_encoder = LabelEncoder()
# Fit the label encoder on the training data labels
area_encoder.fit(y_train)

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
predicted_areas = area_encoder.inverse_transform(y_pred_classes)

# Add the predicted labels to the validation DataFrame
validation_df['Predicted_Area'] = predicted_areas

# Save the validation results DataFrame to an Excel file
validation_results_file_path = 'Validation_Results_With_Predictions.xlsx'
validation_df.to_excel(validation_results_file_path, index=False)

print("Validation results with predictions saved to", validation_results_file_path)
