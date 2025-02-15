import pandas as pd

# Replace the path with the datasets to merge
df1 = pd.read_csv('Datasets/Test_Outpatientdata-1542969243754.csv')
df2 = pd.read_csv('Datasets/Train_Outpatientdata-1542865627584.csv')

# Append the dataframes
appended_df = pd.concat([df1, df2])

# Save the appended dataframe to a csv file with appropriate path
appended_df.to_csv('Datasets/outpatient_appended')