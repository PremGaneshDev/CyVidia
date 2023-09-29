import pandas as pd

# Load the validation dataset and master dataset
validation_file_path = 'Output/SingleFile (1).xlsx'
master_file_path = 'Data/All extracted RBI data - main sheet.xlsx'

validation_df = pd.read_excel(validation_file_path)
master_df = pd.read_excel(master_file_path)

# Initialize a dictionary to store validation results
validation_results = {'Requirement Description': [], 'Validation Result': []}

# Iterate through validation dataset
for index, row in validation_df.iterrows():
    validation_description = row['Requirement Description']  # Use the original description
    
    # Check if the description exists in the master dataset
    if validation_description in master_df['Requirement Description'].values:
        # Find the corresponding area in the master dataset
        master_area = master_df.loc[master_df['Requirement Description'] == validation_description, 'Requirement Area'].values[0]
        
        # Compare with the predicted area
        predicted_area = row['Predicted_Area']
        if predicted_area == master_area:
            validation_result = 'Correct'
        else:
            validation_result = 'Incorrect'
    else:
        validation_result = 'Not Found'
    
    # Store the result in the dictionary
    validation_results['Requirement Description'].append(validation_description)
    validation_results['Validation Result'].append(validation_result)

# Create a DataFrame from the validation results
validation_results_df = pd.DataFrame(validation_results)

# Save the results to an Excel file
validation_results_file_path = 'Output/Validation_Results.xlsx'
validation_results_df.to_excel(validation_results_file_path, index=False)
print("Validation results saved to", validation_results_file_path)
