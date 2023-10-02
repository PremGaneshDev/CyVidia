#importing libraries
import pandas as pd

#creating our own dataframe
first=pd.DataFrame({"req_lab":[1,2,3,4,5,6,7,8,9,10,11],"req_desc":['Hii','Hello','Hell','jumbo','tumbo','apple','ball','cat','dog','how','who']})
second=pd.DataFrame({"desc":['Hell','jumbo','tumbo','apple','Hii','Hello'],"pred":[1,3,2,4,5,6]})

#merging the data
# Merge the DataFrames
merged_data = pd.merge(second, first, left_on='desc', right_on='req_desc', how='left')

# Add a new column based on conditions
merged_data['status'] = 'incorrect'
merged_data.loc[(merged_data['req_lab'] == merged_data['pred']), 'status'] = 'correct'

# Drop unnecessary columns
merged_data = merged_data.drop(['req_lab', 'req_desc'], axis=1)

# Display the final DataFrame
print(merged_data)