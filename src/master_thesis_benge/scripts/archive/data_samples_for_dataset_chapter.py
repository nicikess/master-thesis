import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/ben-ge-train20.csv')

# Replace 'your_column_name' with the actual name of the column you're interested in
column_name = 'tree_cover'

# Find the index of the row with the maximum value in the specified column
max_index = df[column_name].idxmax()

# Get the patch_id from the row with the maximum value
max_tree_cover_patch_id = df.loc[max_index, 'patch_id']

print("The patch_id with the maximum tree_cover:", max_tree_cover_patch_id)
