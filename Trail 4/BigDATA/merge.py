import pandas as pd

# Load your master data set
masterdata_df = pd.read_excel('All extracted RBI data - main sheet.xlsx')

# Load the validation results
validation_results_file_path = 'Validation_Results_With_Predictions.xlsx'
validation_results_df = pd.read_excel(validation_results_file_path)

# Merge the master data set with validation results based on Requirement Description
merged_df = pd.merge(masterdata_df, validation_results_df, on="Requirement Description", how="inner")

# Check if Predicted_Area matches Requirement Area and create a "Correct" column
merged_df['Correct'] = merged_df.apply(lambda row: 'Valid' if row['Predicted_Area'] == row['Requirement Area'] else 'Invalid', axis=1)

# Print the number of correct and incorrect predictions
print(f"Number of correct predictions: {merged_df['Correct'].eq('Valid').sum()}")
print(f"Number of incorrect predictions: {merged_df['Correct'].eq('Invalid').sum()}")
print(f"Total number of predictions: {merged_df.shape[0]}")

# Save the merged results to a new Excel file
merged_results_file_path = 'Merged_Validation_Results.xlsx'
merged_df.to_excel(merged_results_file_path, index=False)

print("Merged validation results saved to", merged_results_file_path)
