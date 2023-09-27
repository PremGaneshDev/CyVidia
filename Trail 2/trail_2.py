import pandas as pd

# Load the Excel file into a DataFrame
excel_file = "POC Based Requirements.xlsx"
df = pd.read_excel(excel_file)

# Create a dictionary to store merged keywords from topics and keywords
merged_keywords_dict = {}
# Create a list of dictionaries with merged keywords from topics and keywords
labels_and_keywords = [{"keywords": keywords, "label": label} for label, keywords in merged_keywords_dict.items()]

# Assign the merged labels to the DataFrame based on keywords in the 'Key Words' column
for item in labels_and_keywords:
    label = item['label']
    keywords = item['keywords']
    keyword_matches = df['Key Words'].apply(lambda x: any(keyword.lower() in x.lower() for keyword in keywords))
    df.loc[keyword_matches, 'Requirements Bucket'] = label

# Save the DataFrame with updated labels to a new Excel file
new_excel_file = "New_Requirements.xlsx"
df.to_excel(new_excel_file, index=False)

# Print a summary of the updated DataFrame
print("Updated DataFrame:")
print(df.head())

# Print a confirmation message
print(f"New Excel file '{new_excel_file}' with updated labels has been created.")
