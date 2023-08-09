import pandas as pd

def get_unique_data(df1, df2, column_name):
    """
    Get the data points that are in df2 but not in df1 based on the specified column_name.

    Parameters:
        df1 (pandas.DataFrame): The first data frame.
        df2 (pandas.DataFrame): The second data frame.
        column_name (str): The name of the column to compare for uniqueness.

    Returns:
        pandas.DataFrame: A new data frame containing the unique data points from df2.
    """
    # Create a set of patch_ids from the first data frame
    df1_ids = set(df1[column_name])

    # Filter the second data frame based on patch_ids that are not in df1_ids
    unique_data = df2[~df2[column_name].isin(df1_ids)]

    return unique_data

def check_new_data_frame(df1, df2, column_name):
    """
    Check if the new data frame (df2) contains any patch_ids from the first data frame (df1).
    
    Parameters:
        df1 (pandas.DataFrame): The first data frame.
        df2 (pandas.DataFrame): The new data frame to be checked.
        column_name (str): The name of the column to compare for uniqueness.
    
    Returns:
        bool: True if df2 contains no patch_ids from df1, False otherwise.
    """
    df1_ids = set(df1[column_name])
    df2_ids = set(df2[column_name])
    
    # Check if there's any intersection between the patch_ids of df1 and df2
    return not bool(df1_ids.intersection(df2_ids))

# Load first data frame
data_frame1 = pd.read_csv('data-splits/ben-ge-train40.csv')
print(len(data_frame1))

data_frame2 = pd.read_csv('data-splits/ben-ge-train60.csv')
print(len(data_frame2))

# Call the function to get the unique data points from data_frame2
result_data_frame = get_unique_data(data_frame1, data_frame2, 'patch_id')

is_valid = check_new_data_frame(data_frame1, result_data_frame, 'patch_id')
print(is_valid)

# Print the result or save it to a new data frame or CSV file as needed
print(len(result_data_frame))

# Save data frame
result_data_frame.to_csv('data-splits/ben-ge-delta-60-40.csv', index=False)
