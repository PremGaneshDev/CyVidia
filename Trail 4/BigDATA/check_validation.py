import pandas as pd 

# Read the validation Excel
validation_df = pd.read_excel('Output/Validation_Results_With_Predictions_And_Score.xlsx')
validation_df = validation_df.drop_duplicates(subset=['Requirement Description'])
print(validation_df.shape)


# Read the master Excel 
master_df = pd.read_excel('Data/All extracted RBI data - main sheet.xlsx')

# Merge the two DataFrames based on Requirement Description matches
merged_df = pd.merge(validation_df, master_df, on='Requirement Description', how='inner')

# Remove duplicates based on a specific column, e.g., 'Requirement Description'
merged_df = merged_df.drop_duplicates(subset=['Requirement Description'])

# # Print all the merged DataFrames
# for index, row in merged_df.iterrows():
#     print(row['Requirement Description'])
#     print("------------------------------------------------------")

print(merged_df.head())
#export to excel
merged_df.to_excel('Output/Merged.xlsx', index=False)

 