# Cell 17
# Assuming you are in the same notebook or script, import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load the saved model
model_path = 'trained_model_rbi_jll_nist'
model = load_model(model_path)

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoders
with open('area_encoder.pickle', 'rb') as handle:
    area_encoder = pickle.load(handle)

with open('bucket_encoder.pickle', 'rb') as handle:
    bucket_encoder = pickle.load(handle)

# Now you can use the loaded model, tokenizer, and encoders to make predictions on new data

# Example: Preprocess new requirement
new_requirement = "Your new requirement text here"
preprocessed_requirement = clean_text(new_requirement)
tokenized_requirement = tokenizer.texts_to_sequences([preprocessed_requirement])
padded_requirement = pad_sequences(tokenized_requirement, maxlen=100)

# Make predictions
predictions = model.predict(padded_requirement)

# Assuming your model outputs probabilities, you can threshold them to get binary predictions
threshold = 0.5
binary_predictions_area = (predictions[0] > threshold).astype(int)
binary_predictions_bucket = (predictions[1] > threshold).astype(int)

# Decode the predictions using the label encoders
area_label = area_encoder.classes_[binary_predictions_area[0]]
bucket_label = bucket_encoder.classes_[binary_predictions_bucket[0]]

# Print the predicted labels
print("Predicted Area Label:", area_label)
print("Predicted Bucket Label:", bucket_label)
